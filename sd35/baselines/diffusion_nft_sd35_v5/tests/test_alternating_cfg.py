"""Gradient, schedule, and real-trainer forward-path regression checks."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from baselines.diffusion_nft_sd35_v5 import cfg_modes as cfg

ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / 'baselines/diffusion_nft/upstream/scripts/train_nft_sd3.py'
CONFIG = ROOT / 'baselines/diffusion_nft_sd35_v5/config'


class AlternatingCFGTests(unittest.TestCase):
    def test_real_autocast_keeps_conditional_grad_after_no_grad_empty(self):
        tree = ast.parse(TRAINER.read_text())
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'alternating_phases'
                    and any(isinstance(x, ast.With) for x in n.body))
        outer = parents[node]
        while not (isinstance(outer, ast.With) and
                   any('torch_autocast' in ast.unparse(item.context_expr) for item in outer.items)):
            outer = parents[outer]
        actual_forward = ast.With(items=copy.deepcopy(outer.items), body=copy.deepcopy(node.body))
        code = compile(ast.fix_missing_locations(ast.Module(body=[actual_forward], type_ignores=[])), '<trainer-autocast>', 'exec')

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
            def forward(self, *, encoder_hidden_states, **kwargs):
                return (self.linear(encoder_hidden_states),)

        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        env = dict(torch=torch, transformer_ddp=model, cfg_modes_extension=cfg,
                   xt=torch.zeros(2, 4), training_timestep=torch.ones(2),
                   negative_embeds=torch.ones(2, 4), embeds=torch.full((2, 4), 2.),
                   negative_pooled_embeds=None, pooled_embeds=None,
                   alternating_phases=('unconditional', 'conditional'),
                   enable_amp=True, mixed_precision_dtype=torch.bfloat16,
                   torch_autocast=lambda **kwargs: torch.autocast('cpu', **kwargs))
        for uncond in (True, False, False):
            env['unconditional_update'] = uncond
            exec(code, env)
            current_c, current_u = env['forward_prediction'], env['current_unconditional_prediction']
            self.assertEqual(current_c.requires_grad, not uncond)
            self.assertEqual(current_u.requires_grad, uncond)
            p, _ = cfg.compose_alternating_predictions(
                phase='unconditional' if uncond else 'conditional',
                current_cond=current_c, current_uncond=current_u,
                old_cond=torch.zeros_like(current_c), old_uncond=torch.zeros_like(current_u),
                center_prediction=torch.zeros_like(current_c), beta=.1,
            )
            p.float().square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            optimizer.zero_grad()

    def test_schedule_includes_zero_and_keeps_fifty_policy_updates(self):
        phases = [cfg.alternating_update_phases(epoch=e, interval=10) for e in range(50)]
        self.assertEqual([i for i, p in enumerate(phases) if len(p) == 2], [0, 10, 20, 30, 40])
        self.assertEqual(sum(p.count('conditional') for p in phases), 50)
        self.assertEqual(sum(p.count('unconditional') for p in phases), 5)
        # A resumed run uses the absolute epoch rather than resetting its cadence.
        self.assertEqual(cfg.alternating_update_phases(epoch=23, interval=10), ('conditional',))
        for epoch, interval in [(-1, 10), (True, 10), (0, 0), (0, 1.5)]:
            with self.assertRaises(ValueError):
                cfg.alternating_update_phases(epoch=epoch, interval=interval)

    def test_exact_predictions_and_gradients(self):
        for phase, positive, negative, cond_grad, empty_grad in [
            ('conditional', 9.4, 4.6, .4, None),
            ('unconditional', 7.4, 6.6, None, .1),
        ]:
            values = [torch.tensor([[x]], requires_grad=True) for x in (10., 6., 1., 2., 7.)]
            c, u, oc, ou, center = values
            p, n = cfg.compose_alternating_predictions(
                phase=phase, current_cond=c, current_uncond=u,
                old_cond=oc, old_uncond=ou, center_prediction=center, beta=.1,
            )
            torch.testing.assert_close(p, torch.tensor([[positive]]))
            torch.testing.assert_close(n, torch.tensor([[negative]]))
            p.sum().backward()
            for x, expected in [(c, cond_grad), (u, empty_grad), (oc, None), (ou, None), (center, None)]:
                if expected is None:
                    self.assertIsNone(x.grad)
                else:
                    torch.testing.assert_close(x.grad, torch.full_like(x, expected))

    def test_trainer_inactive_forward_has_no_graph_and_reads_updated_shared_weights(self):
        tree = ast.parse(TRAINER.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'alternating_phases'
                    and any(isinstance(x, ast.With) for x in n.body))
        code = compile(ast.fix_missing_locations(ast.Module(body=node.body, type_ignores=[])), '<trainer-forward>', 'exec')
        class Model:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(2.))
                self.modes = []
            def __call__(self, *, encoder_hidden_states, **kwargs):
                self.modes.append(torch.is_grad_enabled())
                return (self.weight * encoder_hidden_states,)
        model = Model()
        optimizer = torch.optim.AdamW([model.weight], lr=.01)
        env = dict(torch=torch, transformer_ddp=model, cfg_modes_extension=cfg,
                   xt=torch.zeros(1, 1), training_timestep=torch.ones(1),
                   negative_embeds=torch.ones(1, 1), embeds=torch.full((1, 1), 2.),
                   negative_pooled_embeds=None, pooled_embeds=None)
        for uncond in (True, False):
            env['unconditional_update'] = uncond
            exec(code, env)
            c, u = env['forward_prediction'], env['current_unconditional_prediction']
            self.assertEqual(c.requires_grad, not uncond)
            self.assertEqual(u.requires_grad, uncond)
            torch.testing.assert_close(u, model.weight.detach().reshape(1, 1))
            p, _ = cfg.compose_alternating_predictions(
                phase='unconditional' if uncond else 'conditional', current_cond=c,
                current_uncond=u, old_cond=torch.zeros_like(c), old_uncond=torch.zeros_like(u),
                center_prediction=torch.zeros_like(c), beta=.1,
            )
            p.square().mean().backward()
            self.assertIsNotNone(model.weight.grad)
            optimizer.step(); optimizer.zero_grad()
        self.assertEqual(model.modes, [True, False, False, True])
        self.assertEqual(int(optimizer.state[model.weight]['step']), 2)

    def test_actual_counter_and_ema_guards(self):
        tree = ast.parse(TRAINER.read_text())
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        increments = [n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
                      and ast.unparse(n.target) == 'global_step']
        self.assertEqual(len(increments), 1)
        self.assertEqual(ast.unparse(parents[increments[0]].test), 'not unconditional_update')
        ema_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and ast.unparse(n.func) == 'ema.step')
        guard = parents[parents[ema_call]]
        self.assertIn('not unconditional_update', ast.unparse(guard.test))


if __name__ == '__main__':
    unittest.main()
