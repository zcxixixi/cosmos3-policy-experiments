"""Twenty-four compact matched-name q0 predictions; no physics.

--output BASE creates only absent BASE/matched-names. The actual pipeline
tokenizer and packed inputs must verify milk box / cream cheese: two target
IDs, eleven prompt IDs, equal UND122/joint388 and equal target positions.
X12/X15 x V195/V198 x two names x native/all_allowed/full_hardmask.
720 model forwards, 70200 official attention dispatches, 48 original draws.
Raw decoder boundaries are full37 only at t0; later steps save boundary36
action only and live byte hashes for omitted arrays. One t0/L35 full QKV
capture plus all30 actual L35 action-Q/target-KV/returned-action fragments.
Eight native cases additionally retain actual target-start-to-UND-end K/V.
No extra attention replay, 135-site export or MLP capture.
The entire new stage, including records and metadata, must fit within 2.5 GiB.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback


ROOT = Path('/home/current/work/cosmos3')
LANGUAGE_SHA = '74ac42db450a1a7cd308c837bf75049cd2dcae858a4869dab1f0f60450c69b13'
LANGUAGE_COMPLETE_SHA = '43c64a98a2c2423155b4da6e4409c331d6c48f65c6138ac0d2f9f512619bfae7'
POSITIONS_SHA = 'c9cc366e002d8c89ab9c779af35b0191c4ec6d56a157da77499914c6b8605146'
STAGE = 'matched-names'
SCENES = dict(x12=12, x15=15)
PROMPTS = dict(milk_box='pick up the milk box and place it in the basket',
               cream_cheese='pick up the cream cheese and place it in the basket')
TARGET_IDS = dict(milk_box=[14074, 3745], cream_cheese=[12644, 17163])
TARGETS = dict(milk_box='milk_1', cream_cheese='cream_cheese_1')
ARMS = ('native', 'all_allowed', 'full_hardmask')
PLAN = tuple((f'{scene}_{goal}_V{v}_A195_{arm}', scene, goal, v, arm)
             for arm in ARMS for scene in SCENES for goal in PROMPTS for v in (195, 198))
WINDOW = dict(steps=[0, 14], layers=[0, 8])
MAX_BYTES = 5 * 1024 ** 3 // 2
MIN_FREE_BYTES = MAX_BYTES + 2 * 1024 ** 3
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'language-contract.pt',
         'dispatch-captures.json')
DISPATCH_KEYS = ('native_UND_dispatch', 'native_GEN_dispatch', 'all_allowed_dispatch', 'cut_dispatch')
SCOPE = ('Twenty-four fixed natural predictions at X12/X15 with actual equal-length names milk box/cream cheese, '
         'V195/V198 and A/runtime195. Scene task7 remains the original milk basket task; naming success is not '
         'a repair of the original milk instruction at 15cm. Four new milk-box native predictions are first '
         'observations, not independently repeated controls. X12 cream-cheese six cases reproduce the prior '
         'complete language stage. Native X15 uses its own real restored image and initial state; X12 records '
         'supply noise/schedule only. All masked sites share union66 all-allowed arithmetic; only t0..14/L0..8 '
         'current50 queries block future200 keys. Hard masks also redistribute remaining weights. Sparse raw '
         'capture is explicit: omitted arrays have live byte hashes, not offline raw equality. No head or MLP '
         'semantic claim, attention replay, training, solver replay, environment or physical execution.')
PRIMARY = dict(name='H_matched_name_native_targets_x15',
    cases=[label for label, scene, _, _, arm in PLAN if scene == 'x15' and arm == 'native'],
    requested_objects=TARGETS, criterion='Each X15 native V195/V198 case must strictly select only its named object.',
    all24_physics_required=True, numerical_sham_is_separate_cut_attribution_diagnostic=True,
    no_q0_score_or_rank_selection=True, original_milk_15cm_goal_not_declared_repaired=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_counts(arm=None):
    if arm is None:
        rows = [expected_counts(item[-1]) for item in PLAN]
        return {key: sum(row[key] for row in rows) for key in rows[0]}
    assert arm in ARMS
    return dict(model_before=30, model_after=30, block_before=1080, block_after=1080, pre_W=1080,
        native_UND_dispatch=1080, native_GEN_dispatch=1080,
        all_allowed_dispatch=0 if arm == 'native' else 1080,
        cut_dispatch=135 if arm == 'full_hardmask' else 0,
        substituted_sites=0 if arm == 'native' else 1080)


def disk_bytes(folder):
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())


def source_contract(baseline):
    """CPU read-only provenance/typed-input checks before output or model loading."""
    lp, pp = (ROOT / 'work' / name for name in
              ('probe_cosmos_milk_language_selectivity.py', 'probe_cosmos_milk_future_current_positions.py'))
    assert sha(lp) == LANGUAGE_SHA and sha(pp) == POSITIONS_SHA
    language, positions = load('matched_names_language', lp), load('matched_names_positions', pp)
    context, position_context = language.source_contract(baseline), positions.source_contract(baseline)
    assert context['frozen_contract'] == position_context['frozen_contract']
    assert context['original_noise_sources'] == position_context['original_noise_sources']
    for key in context['paths']:
        assert str(context['paths'][key]) == str(position_context['paths'][key])
    old = baseline / language.STAGE
    assert not (old / 'failed.json').exists() and sha(old / 'complete.json') == LANGUAGE_COMPLETE_SHA
    complete, results = read(old / 'complete.json'), read(old / 'results.json')
    assert complete['state'] == results['state'] == 'complete' and complete['script_sha256'] == LANGUAGE_SHA
    documents = {'complete': LANGUAGE_COMPLETE_SHA}
    for name in ('results', 'provenance', 'protocol', 'sources', 'primary-prediction', 'primary-language-contract'):
        actual = sha(old / (name + '.json'))
        assert actual == complete[name.replace('-', '_') + '_sha256']
        documents[name] = actual
    assert complete['fresh_q0_predictions'] == 18 and complete['fresh_model_forwards'] == 540
    assert complete['official_dispatch_counts'] == language.expected_counts()
    assert complete['total_official_dispatch_calls'] == 52650
    references = {}
    by_case = {row['case']: row for row in results['cases']}
    for arm in ARMS:
        for v in (195, 198):
            label = f'cream_cheese_V{v}_A195_{arm}'
            row, folder = by_case[label], old / label
            assert row['goal'] == 'cream_cheese' and row['arm'] == arm and row['actual_und_len'] == 122
            assert set(row['files_sha256']) == set(language.FILES)
            for name, expected in row['files_sha256'].items():
                assert sha(folder / name) == expected
            assert len(row['boundary_files']) == 30
            for step, item in enumerate(row['boundary_files']):
                assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
                assert sha(folder / item['file']) == item['sha256']
            references[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'])
    x12 = baseline / 'position-threshold/inputs/x12'
    assert Path(context['image']).read_bytes() == (x12 / 'input.png').read_bytes()
    im = read(x12.parent / 'metadata.json')
    assert im['shifts_cm']['x12'] == 12 and im['task_index'] == 7 and im['additional_warmup_steps'] == 0
    names = tuple(path.name for path in x12.iterdir() if path.is_file())
    assert {'state.npy', 'arrays.json', 'controller.npz', 'objects.npz', 'observations.npz', 'input.png'} <= set(names)
    context['scenes'] = dict(x12=dict(folder=str(x12), image=str(x12 / 'input.png'), shift_cm=12,
        files_sha256={name: sha(x12 / name) for name in sorted(names)}, physical_scene=im['scenes']['x12'],
        input_root=str(x12.parent), metadata_file=str(x12.parent / 'metadata.json'), complete_file=str(x12.parent / 'complete.json'),
        inputs_metadata_path=str(x12.parent / 'metadata.json'), inputs_metadata_sha256=sha(x12.parent / 'metadata.json'),
        hash_scope='Typed threshold scene verified by frozen source contract; file hashes newly bound by this preflight.'),
        x15=dict(position_context['scenes']['x15'], input_root=str(position_context['input_root']),
            metadata_file=str(position_context['input_root'] / 'metadata.json'), complete_file=str(position_context['input_root'] / 'complete.json'),
            inputs_metadata_path=str(position_context['input_root'] / 'metadata.json'),
            inputs_metadata_sha256=sha(position_context['input_root'] / 'metadata.json')))
    context['modules']['language'], context['modules']['positions'] = language, positions
    context['canonical_references'] = references
    context['source_evidence']['matched_name_sources'] = dict(language_script_sha256=LANGUAGE_SHA,
        language_documents_sha256=documents, canonical_x12_references=references,
        positions_script_sha256=POSITIONS_SHA, position_source_evidence=position_context['source_evidence'])
    context['out'] = baseline / STAGE
    return context


def prepared(baseline):
    return source_contract(baseline)


def target_span(kwargs, goal, prompt_ids, language, runtime):
    ids = runtime.cpu(kwargs['input_ids']).flatten().tolist()
    start = language.unique_span(ids, prompt_ids, 'actual packed matched-name prompt/' + goal)
    target = language.unique_span(prompt_ids, TARGET_IDS[goal], 'actual matched-name target/' + goal)
    positions = [start + target, start + target + 1]
    assert len(prompt_ids) == 11 and [ids[i] for i in positions] == TARGET_IDS[goal]
    assert int(kwargs['und_len']) == 122 and int(kwargs['sequence_length']) == 388
    return dict(goal=goal, prompt=PROMPTS[goal], input_ids=runtime.cpu(kwargs['input_ids']),
        prompt_token_ids=list(prompt_ids), prompt_span=[start, start + 11], target_token_ids=TARGET_IDS[goal],
        target_positions=positions, und_len=122, sequence_length=388,
        text_indexes=runtime.cpu(kwargs['text_indexes']), position_ids=runtime.cpu(kwargs['position_ids']))


def observer_class(frozen, language):
    class MatchedObserver(frozen.FutureReadCuts):
        def __init__(self, *args, scene, goal, prompt_ids, native_info, legacy_reference, **kwargs):
            super().__init__(*args, **kwargs)
            self.scene, self.goal, self.prompt_ids = scene, goal, prompt_ids
            self.native_info, self.legacy_reference = native_info, legacy_reference
            self.language, self.actual_kwargs, self.legacy_boundary = None, None, None
            self.und_boundaries, self.hidden_metadata, self.dispatch_capture = [], [], None
            self.dispatch_files = []
            self.finite_boundary_checks = 0

        def meta(self, value):
            return language.tensor_meta(value, self.runtime)

        def before_model(self, module, args, kwargs):
            assert not args and not self.in_model and self.active is None
            self.step += 1
            assert self.step < 30 and kwargs['return_dict'] is False
            actual = dict(layout=self.components._layout(kwargs),
                metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.components._STRUCTURAL_KEYS})
            packing = {'input_ids', 'und_len', 'sequence_length', 'text_indexes', 'position_ids',
                       'vision_sequence_indexes', 'vision_mse_loss_indexes', 'action_sequence_indexes', 'action_mse_loss_indexes'}
            for key in self.components._STRUCTURAL_KEYS:
                if key not in packing:
                    self.runtime.require_exact(actual['metadata'][key], self.cache['steps'][self.step]['metadata'][key],
                                               'frozen structural/schedule/domain only/' + key)
            if self.native_info is not None:
                self.runtime.require_exact(actual, self.native_info['actual_pack'][self.step], 'own scene/goal/V all30 pack')
            derived = language.dynamic_indexes(kwargs, self.model, self.components, self.runtime)
            span = target_span(kwargs, self.goal, self.prompt_ids, language, self.runtime)
            if self.index is None:
                self.index, self.language = derived, span
                target_keys = self.runtime.cpu(kwargs['text_indexes']).flatten()[span['target_positions']]
                self.runtime.require_exact(target_keys, self.torch.tensor(span['target_positions']), 'actual target ID to UND key mapping')
                self.runtime.require_exact(self.runtime.cpu(kwargs['text_indexes']).flatten(),
                    self.torch.arange(122), 'actual whole text-to-UND key mapping for causal suffix')
                self.masks = language.dynamic_masks(derived, self.torch)
                self.gpu_masks = {name: item['mask'].to(kwargs['vision_tokens'][0].device) for name, item in self.masks.items()}
                self.first = self.runtime.cpu(kwargs)
                self.clean_current = self.first['vision_tokens'][0][0, :, 0].clone()
                if self.native_info is not None:
                    self.runtime.require_exact(self.first, self.native_info['first_model_kwargs'], 'own scene/goal/V entire initial kwargs')
                    self.runtime.require_exact(self.clean_current, self.native_info['actual_clean_current'], 'own scene current condition')
                self.torch.save(dict(indexes=self.index, masks=self.masks), self.folder / 'indexes-and-masks.pt')
            else:
                self.runtime.require_exact(derived, self.index, 'all30 derived same packing')
                self.runtime.require_exact(span, self.language, 'all30 actual name/position IDs')
            self.runtime.require_exact(self.runtime.cpu(kwargs['vision_tokens'][0][0, :, 0]), self.clean_current, 'actual own-scene current clamp')
            assert self.torch.count_nonzero(kwargs['action_tokens'][0][:, 10:]) == 0
            self.actual_kwargs = self.runtime.cpu(kwargs)
            if self.legacy_reference is not None:
                item = self.legacy_reference['boundary_files'][self.step]
                self.legacy_boundary = self.torch.load(Path(self.legacy_reference['folder']) / item['file'],
                    map_location='cpu', weights_only=True, mmap=True)
                assert language.tree_sha(self.actual_kwargs, self.runtime) == self.legacy_boundary['actual_model_kwargs_sha256']
                if self.legacy_boundary['actual_model_kwargs'] is not None:
                    self.runtime.require_exact(self.actual_kwargs, self.legacy_boundary['actual_model_kwargs'], 'canonical actual full saved kwargs')
            self.calls.append(actual)
            self.counts['model_before'] += 1
            self.in_model, self.next_layer = True, 0
            self.boundaries, self.events, self.und_boundaries, self.hidden_metadata = [], [], [], []

        def observe_boundary(self, und, gen):
            torch = self.torch
            assert und.shape == (122, 4096) and gen.shape == (266, 4096)
            assert und.dtype == gen.dtype == torch.bfloat16
            current = gen.index_select(0, self.index['current_rows'].to(gen.device))
            action = gen.index_select(0, self.index['action_rows'].to(gen.device))
            assert all(bool(torch.isfinite(item).all()) for item in (und, current, action))
            b = len(self.hidden_metadata)
            self.hidden_metadata.append(dict(boundary=b, und=self.meta(und), current=self.meta(current), action=self.meta(action)))
            if self.step == 0:
                self.und_boundaries.append(self.runtime.cpu(und))
                self.boundaries.append(dict(current=self.runtime.cpu(current), action=self.runtime.cpu(action)))
            elif b == 36:
                self.boundaries.append(dict(action=self.runtime.cpu(action)))
            self.finite_boundary_checks += 1

        def before_block(self, layer, module, args):
            assert self.in_model and self.active is None and layer == self.next_layer and len(args) == 3
            if layer == 0:
                self.observe_boundary(args[0], args[1])
            self.active, self.pending, self.site_calls = layer, None, []
            self.counts['block_before'] += 1

        def dispatch(self, *args, **kwargs):
            torch = self.torch
            assert self.in_model and self.active is not None
            assert len(args) == 3 and set(kwargs) == {'is_causal', 'enable_gqa', 'backend', 'parallel_config'}
            assert kwargs['enable_gqa'] is True and kwargs['backend'] is kwargs['parallel_config'] is None
            causal = kwargs['is_causal']
            assert type(causal) is bool and self.site_calls == ([] if causal else ['UND'])
            self.site_calls.append('UND' if causal else 'GEN')
            qlen, kvlen = (122, 122) if causal else (266, 388)
            assert args[0].shape == (1, qlen, 32, 128) and args[1].shape == args[2].shape == (1, kvlen, 8, 128)
            assert all(item.dtype == torch.bfloat16 and bool(torch.isfinite(item).all()) for item in args)
            witnesses = tuple(item.clone(memory_format=torch.preserve_format) for item in args)
            rng = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(args[0].device).clone())
            self.counts['native_UND_dispatch' if causal else 'native_GEN_dispatch'] += 1
            native = self.original(*args, **kwargs)
            assert native.shape == (1, qlen, 32, 128) and native.dtype == torch.bfloat16 and bool(torch.isfinite(native).all())
            if causal:
                returned = native
            else:
                current, action, future, union = (self.index[key].to(native.device) for key in
                                                  ('current_rows', 'action_rows', 'future_rows', 'union_rows'))
                active = self.arm == 'full_hardmask' and self.step <= 14 and self.active <= 8
                event = dict(step=self.step, layer_zero_based=self.active, arm=self.arm, selected_window_cut_active=active,
                    original_GEN_dispatch_calls=1, original_UND_dispatch_calls=1, extra_all_allowed_calls=0,
                    extra_cut_calls=0, to_add_out_shape=[266, 4096], to_add_out_actual_calls=0, processor_recompute_calls=0)
                allowed = cut = None
                if self.arm == 'native':
                    returned = native
                else:
                    self.counts['all_allowed_dispatch'] += 1
                    allowed = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['all_allowed']))
                    assert allowed.shape == native.shape and allowed.dtype == native.dtype and bool(torch.isfinite(allowed).all())
                    returned = native.clone()
                    returned[:, union] = allowed[:, union]
                    event['extra_all_allowed_calls'] = 1
                    if active:
                        self.counts['cut_dispatch'] += 1
                        cut = self.original(*args, **dict(kwargs, attn_mask=self.gpu_masks['cut_future_current']))
                        assert cut.shape == native.shape and cut.dtype == native.dtype and bool(torch.isfinite(cut).all())
                        unblocked = self.masks['cut_future_current']['unblocked_rows'].to(native.device)
                        assert frozen.byte_exact(cut[:, unblocked], allowed[:, unblocked], torch)
                        returned[:, current] = cut[:, current]
                        event.update(extra_cut_calls=1, all_unblocked_GEN_rows_vs_same_input_all_allowed_byte_exact=True)
                    assert frozen.byte_exact(returned[:, action], allowed[:, action], torch)
                    assert frozen.byte_exact(returned[:, future], native[:, future], torch)
                    self.counts['substituted_sites'] += 1
                event.update(substituted_query_rows=0 if self.arm == 'native' else 66,
                    future200_native_bytes_preserved=True, UND_dispatch_return_unmodified=True)
                self.events.append(event)
                self.pending = returned.squeeze(0).flatten(-2, -1)
                if self.active == 35:
                    assert self.dispatch_capture is None
                    full = self.step == 0
                    qcpu, kcpu, vcpu = (self.runtime.cpu(item) if full else None for item in args)
                    native_cpu = self.runtime.cpu(native) if full else None
                    returned_cpu = (native_cpu if self.arm == 'native' else self.runtime.cpu(returned)) if full else None
                    assert action.detach().cpu().tolist() == list(range(250, 266))
                    target = self.torch.tensor(self.language['target_positions'], device=args[1].device)
                    suffix_start, suffix_end = self.language['target_positions'][0], self.index['und_len']
                    suffix_indexes = self.torch.arange(suffix_start, suffix_end)
                    suffix_saved = self.arm == 'native'
                    suffix_k = (kcpu[:, suffix_start:suffix_end] if full else
                        self.runtime.cpu(args[1][:, suffix_start:suffix_end])) if suffix_saved else None
                    suffix_v = (vcpu[:, suffix_start:suffix_end] if full else
                        self.runtime.cpu(args[2][:, suffix_start:suffix_end])) if suffix_saved else None
                    self.dispatch_capture = dict(step=self.step, layer_zero_based=35, scene=self.scene, goal=self.goal, arm=self.arm,
                        Q=qcpu, K=kcpu, V=vcpu, native_output=native_cpu, actual_returned_output=returned_cpu,
                        Qaction16=qcpu[:, 250:266] if full else self.runtime.cpu(args[0][:, action]),
                        Ktarget2=kcpu[:, suffix_start:suffix_start + 2] if full else self.runtime.cpu(args[1].index_select(1, target)),
                        Vtarget2=vcpu[:, suffix_start:suffix_start + 2] if full else self.runtime.cpu(args[2].index_select(1, target)),
                        Znative_action16=native_cpu[:, 250:266] if full else self.runtime.cpu(native[:, action]),
                        Zmerged_action16=(None if self.arm == 'native' else
                            returned_cpu[:, 250:266] if full else self.runtime.cpu(returned[:, action])),
                        target_span=self.language, target_key_indexes=self.torch.tensor(self.language['target_positions']),
                        causal_suffix_key_indexes=suffix_indexes, causal_suffix_raw_saved=suffix_saved,
                        Kcausal_suffix=suffix_k, Vcausal_suffix=suffix_v,
                        native_kwargs=dict(kwargs), QKV_metadata=[self.meta(item) for item in args],
                        native_output_metadata=self.meta(native), actual_returned_output_metadata=self.meta(returned),
                        sparse_metadata=dict(Qaction16=self.meta(args[0][:, action]),
                            Ktarget2=self.meta(args[1].index_select(1, target)), Vtarget2=self.meta(args[2].index_select(1, target)),
                            Znative_action16=self.meta(native[:, action]),
                            Zmerged_action16=None if self.arm == 'native' else self.meta(returned[:, action]),
                            Kcausal_suffix=self.meta(args[1][:, suffix_start:suffix_end]) if suffix_saved else None,
                            Vcausal_suffix=self.meta(args[2][:, suffix_start:suffix_end]) if suffix_saved else None),
                        indexes=self.index, full_QKV_raw_saved=full,
                        raw_scope='t0 full-GEN QKV/output; all30 actual action16 Q, target2 K/V, native action output and nonnative merged action output; native only additionally target-start-to-UND-end causal suffix K/V; other full raw omitted with live SHA; no replay.')
            for actual, witness in zip(args, witnesses):
                assert frozen.byte_exact(actual, witness, torch), 'Original/extra official dispatch changed actual QKV'
            assert frozen.byte_exact(rng['cpu'], torch.get_rng_state(), torch)
            assert frozen.byte_exact(rng['cuda'], torch.cuda.get_rng_state(args[0].device), torch)
            assert bool(torch.isfinite(returned).all())
            return returned

        def before_W(self, layer, module, args):
            super().before_W(layer, module, args)
            if layer == 35:
                if self.step == 0:
                    actual = self.runtime.cpu(args[0])
                    saved = self.dispatch_capture['actual_returned_output'].squeeze(0).flatten(-2, -1)
                    self.runtime.require_exact(actual, saved, 'actual L35 pre-W flatten')
                    # The original observed pre-W bytes are exact; reuse that already
                    # saved storage rather than exporting a second equal 2MiB array.
                    self.dispatch_capture['actual_pre_W'] = saved
                else:
                    self.dispatch_capture['actual_pre_W'] = None
                self.dispatch_capture['actual_pre_W_metadata'] = self.meta(args[0])

        def after_block(self, layer, module, args, output):
            assert self.active == layer and self.pending is None and self.site_calls == ['UND', 'GEN']
            assert type(output) is tuple and len(output) == 2
            self.observe_boundary(output[0], output[1])
            self.active, self.next_layer = None, layer + 1
            self.counts['block_after'] += 1

        def after_model(self, module, args, output):
            assert self.in_model and self.active is None and self.next_layer == 36
            assert len(self.events) == 36 and len(self.hidden_metadata) == 37
            assert type(output) is tuple and len(output) == 3 and type(output[2]) is list and len(output[2]) == 1
            assert output[2][0].shape == (16, 64) and bool(self.torch.isfinite(output[2][0]).all())
            action = self.torch.stack([item['action'] for item in self.boundaries]) if self.step == 0 else self.boundaries[0]['action']
            current = self.torch.stack([item['current'] for item in self.boundaries]) if self.step == 0 else None
            und = self.torch.stack(self.und_boundaries) if self.step == 0 else None
            actual_output = self.runtime.cpu(output)
            if self.legacy_boundary is not None:
                assert language.tree_sha(actual_output, self.runtime) == self.legacy_boundary['model_output_sha256']
                self.runtime.require_exact(action if self.step == 0 else action,
                    self.legacy_boundary['action_hidden'] if self.step == 0 else self.legacy_boundary['action_hidden'][36], 'canonical action raw saved scope')
                if self.step == 0:
                    for key, item in (('current_hidden', current), ('und_hidden', und), ('model_output', actual_output)):
                        self.runtime.require_exact(item, self.legacy_boundary[key], 'canonical t0 complete actual/' + key)
            frame_meta = self.meta(self.actual_kwargs['vision_tokens'][0][0, :, 0])
            data = dict(step=self.step, scene=self.scene, goal=self.goal, case_arm=self.arm,
                action_hidden=action, action_boundary_ids=list(range(37)) if self.step == 0 else [36],
                current_hidden=current, und_hidden=und, hidden_boundary_metadata=self.hidden_metadata,
                actual_model_kwargs=self.actual_kwargs if self.step == 0 else None, model_output=actual_output if self.step == 0 else None,
                actual_model_kwargs_sha256=language.tree_sha(self.actual_kwargs, self.runtime),
                model_output_sha256=language.tree_sha(actual_output, self.runtime),
                actual_model_action_tokens=self.actual_kwargs['action_tokens'][0], actual_model_action_output=actual_output[2][0],
                actual_model_action_timesteps=self.actual_kwargs['action_timesteps'],
                actual_current_frame_metadata=frame_meta, actual_current_frame_sha256=frame_meta['sha256'],
                solver_sigma_from_same_frozen_schedule=self.source['sigmas'][self.step].clone(),
                actual_pack=self.calls[-1], site_events=self.events,
                raw_scope=dict(full37_UND_current_action='t0 only', action_hidden='t0 [37,16,4096]; other29 [16,4096] actual boundary36',
                    full_kwargs_and_tuple='t0 only; other29 live full-tree byte SHA', omitted_hidden='live metadata at all37 boundaries; no offline raw equality'),
                boundary_convention='0 actual decoder input; 1..36 actual complete block outputs; boundary36 before final norm.')
            path = self.folder / f'boundaries-t{self.step:02d}.pt'
            self.torch.save(data, path)
            self.files.append(dict(step=self.step, file=path.name, sha256=sha(path), bytes=path.stat().st_size,
                action_shape=list(action.shape), action_boundary_ids=data['action_boundary_ids'],
                current_shape=None if current is None else list(current.shape), und_shape=None if und is None else list(und.shape),
                actual_model_kwargs_sha256=data['actual_model_kwargs_sha256'], model_output_sha256=data['model_output_sha256']))
            assert self.dispatch_capture is not None and self.dispatch_capture['step'] == self.step
            capture_path = self.folder / f'dispatch-t{self.step:02d}-L35.pt'
            self.torch.save(self.dispatch_capture, capture_path)
            self.dispatch_files.append(dict(step=self.step, layer_zero_based=35, file=capture_path.name,
                sha256=sha(capture_path), bytes=capture_path.stat().st_size, full_QKV_raw_saved=self.step == 0))
            self.outputs.append(actual_output[2][0])
            self.counts['model_after'] += 1
            self.in_model = False
            self.boundaries, self.events, self.und_boundaries, self.hidden_metadata = [], [], [], []
            self.actual_kwargs, self.legacy_boundary, self.dispatch_capture = None, None, None

        def report(self, record):
            assert self.restored and not self.handles and not self.in_model and self.active is None
            assert self.counts == expected_counts(self.arm) and self.finite_boundary_checks == 1110
            assert len(self.calls) == len(self.files) == len(self.outputs) == 30
            self.runtime.require_exact(self.torch.stack(self.outputs), record['action_velocity'], 'all30 actual tuple action equals head')
            assert self.reader.settings(self.torch, self.model.layers[0].self_attn.processor) == self.settings
            assert self.dispatch_capture is None and len(self.dispatch_files) == 30
            manifest = dict(state='complete', files=self.dispatch_files, full_QKV_raw_steps=[0],
                all30_sparse_shapes=dict(Qaction16=[1, 16, 32, 128], Ktarget2=[1, 2, 8, 128],
                    Vtarget2=[1, 2, 8, 128], Znative_action16=[1, 16, 32, 128]),
                Zmerged_action16='Same shape for masked arms; native null because native output is already saved.',
                native_only_all30_causal_suffix_KV_saved=self.arm == 'native',
                causal_suffix_key_indexes=list(range(self.language['target_positions'][0], self.index['und_len'])),
                full_other_QKV_output_raw_omitted_with_live_SHA=True, no_reconstruction_or_replay=True, scope=SCOPE)
            write(self.folder / 'dispatch-captures.json', manifest, exclusive=True)
            return dict(state='complete', scene=self.scene, goal=self.goal, arm=self.arm, counts=self.counts,
                boundary_files=self.files, full37_boundary_raw_steps=[0], final_action_boundary_raw_steps=list(range(30)),
                full_model_kwargs_and_tuple_raw_steps=[0], full_QKV_raw_sites=1,
                all30_actual_L35_sparse_Qaction_target_KV_and_action_output_saved=True,
                native_only_all30_actual_L35_causal_suffix_KV_saved=self.arm == 'native',
                dispatch_files=self.dispatch_files, dispatch_capture_manifest_sha256=sha(self.folder / 'dispatch-captures.json'),
                indexes_and_masks_sha256=sha(self.folder / 'indexes-and-masks.pt'),
                all_actual_dispatch_return_flatten_equals_pre_W_byte_exact=True,
                all1080_actual_full266_projection_calls_preserved=True,
                all_unblocked_GEN_rows_same_input_allallowed_exact=self.arm == 'full_hardmask',
                future200_and_UND_native_returns_preserved=True, all37_by30_observed_hidden_finite=True,
                all_original_QKV_and_CPU_CUDA_global_RNG_bytes_preserved=True,
                extra_model_attention_replay_projection_MLP_calls=0, site_raw_chain_captures=0,
                all_owned_hooks_removed=True, original_dispatch_symbol_restored=True, scope=SCOPE)
    return MatchedObserver


def input_checks(record, observer, originals, v, runtime):
    torch, p = runtime.torch, record['prepared_latents_and_masks']
    assert type(p) is tuple and len(p) == 12 and p[0].shape == (1, 48, 5, 10, 20) and p[0].dtype == torch.float32
    runtime.require_exact(p[0][:, :, 1:], originals[v]['prepared_latents_and_masks'][0][:, :, 1:], 'raw selected V future preparation')
    runtime.require_exact(p[2], originals[195]['prepared_latents_and_masks'][2], 'raw A195 whole preparation')
    for i in range(12):
        if i not in (0, 2):
            runtime.require_exact(p[i], originals[195]['prepared_latents_and_masks'][i], 'other ten preparation fields')
    runtime.require_exact(p[0][0, :, 0].to(torch.bfloat16), observer.clean_current, 'own scene prepared current to BF16')
    runtime.require_exact(record['model_input'], observer.first, 'inner/outer actual initial kwargs')
    runtime.require_exact(record['model_input']['action_tokens'], originals[195]['model_input']['action_tokens'], 'A195 initial action input')
    runtime.require_exact(record['action_states'][0], originals[195]['action_states'][0], 'A195 actual FP32 solver initial sample')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(record[key], originals[195][key], 'complete source schedule/' + key)
    assert p[3] == 20 and p[8].tolist() == [5] and p[10] == 10 and torch.count_nonzero(p[7]) == 0
    runtime.require_exact(p[5].flatten(), torch.tensor([1., 0., 0., 0., 0.], dtype=p[5].dtype), 'condition mask')
    assert torch.count_nonzero(p[2][:, 10:]) == torch.count_nonzero(record['action_states'][..., 10:]) == 0
    for step, item in enumerate(observer.files):
        data = torch.load(observer.folder / item['file'], map_location='cpu', weights_only=True, mmap=True)
        runtime.require_exact(data['actual_model_action_tokens'], record['action_states'][step, 0].to(torch.bfloat16), 'actual all30 solver cast')
        runtime.require_exact(data['actual_model_action_output'], record['action_velocity'][step], 'actual all30 tuple action')
        runtime.require_exact(data['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'actual all30 sigma')
        assert data['actual_current_frame_sha256'] == observer.meta(observer.clean_current)['sha256']
    return dict(state='exact', own_scene_current_clamp_preparation_and_all30_model_frames=True,
        entire_selected_V_future_and_A195_preparation_byte_exact=True, all30_solver_casts_tuple_actions_sigma_byte_exact=True,
        all31_FP32_solver_padding54_zero=True, other_ten_prepared_fields_exact=True,
        full_initial_kwargs_equal_own_scene_goal_V_native=observer.native_info is not None,
        native_first_observation=observer.arm == 'native', independent_new_native_repeat_performed=False)


def matched_contract(infos, runtime):
    rows, torch = [], runtime.torch
    for scene in SCENES:
        for v in (195, 198):
            a, b = (infos[(scene, goal, v)] for goal in PROMPTS)
            left, right = a['first_model_kwargs'], b['first_model_kwargs']
            assert left.keys() == right.keys()
            for key in left:
                if key != 'input_ids':
                    runtime.require_exact(left[key], right[key], 'actual matched names entire initial kwargs/' + key)
            changed = torch.nonzero(left['input_ids'].flatten() != right['input_ids'].flatten()).flatten().tolist()
            assert changed == a['language']['target_positions'] == b['language']['target_positions'] and len(changed) == 2
            for goal, info, value in zip(PROMPTS, (a, b), (left, right)):
                assert [int(value['input_ids'].flatten()[i]) for i in changed] == TARGET_IDS[goal]
                assert info['language']['und_len'] == 122 and info['language']['sequence_length'] == 388
            for first, second in zip(a['actual_pack'], b['actual_pack']):
                runtime.require_exact(first['layout'], second['layout'], 'matched all30 layout')
                for key in first['metadata']:
                    if key != 'input_ids':
                        runtime.require_exact(first['metadata'][key], second['metadata'][key], 'matched all30 positions/structure/' + key)
            rows.append(dict(scene=scene, vision_noise_source_seed=v, changed_actual_input_id_positions=changed,
                actual_und_len=122, actual_joint_length=388, complete_initial_kwargs_only_two_target_IDs_changed=True,
                all30_other_pack_fields_and_positions_byte_exact=True))
        for goal in PROMPTS:
            runtime.require_exact(infos[(scene, goal, 195)]['actual_pack'], infos[(scene, goal, 198)]['actual_pack'], 'own scene/goal cross-V structural pack')
    return dict(state='exact', cases=rows, goals=list(PROMPTS), target_token_ids=TARGET_IDS,
        actual_prompt_token_lengths=dict.fromkeys(PROMPTS, 11), full_physics_required=True, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-source-contract', action='store_true')
    args = parser.parse_args()
    baseline = args.output.resolve()
    context = source_contract(baseline)
    available_before_stage = shutil.disk_usage(baseline).free
    assert available_before_stage >= MIN_FREE_BYTES, 'Pre-forward free disk must cover 2.5GiB stage plus 2GiB reserve'
    if args.check_source_contract:
        print(json.dumps(dict(state='source_contract_verified', stage=STAGE, scenes=context['scenes'],
            expected_counts=expected_counts(), total_official_dispatch_calls=70200, model_forwards=720,
            available_disk_bytes=available_before_stage, maximum_stage_bytes=MAX_BYTES, minimum_free_bytes=MIN_FREE_BYTES), indent=2))
        return
    out, paths, contract = context['out'], context['paths'], context['frozen_contract']
    assert not out.exists(), 'Absent output required; failed/partial stage is never retried'
    out.mkdir(exist_ok=False)
    labels = [item[0] for item in PLAN]
    protocol = dict(plan=[dict(case=label, scene=scene, goal=goal, prompt=PROMPTS[goal], arm=arm,
        vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
        shift_cm=SCENES[scene], window=WINDOW if arm == 'full_hardmask' else None) for label, scene, goal, v, arm in PLAN],
        trial_order=labels, prompts=PROMPTS, target_objects=TARGETS, scenes=context['scenes'],
        selected_window_zero_based_inclusive=WINDOW, primary_hypothesis=PRIMARY, q0_predictions=24, model_forwards=720,
        expected_counts=expected_counts(), original_dispatch_calls=51840, extra_all_allowed_dispatch_calls=17280,
        selected_extra_cut_dispatch_calls=1080, total_official_dispatch_calls=70200, original_randn_calls=48,
        full37_UND_current_action_raw_steps=[0], other29_action_hidden_raw_shape=[16, 4096], other29_action_hidden_boundary=36,
        full_kwargs_and_model_tuple_raw_steps=[0], all30_action_input_output_timestep_sigma_raw=True,
        full_QKV_raw_site=dict(step=0, layer_zero_based=35, sites_per_case=1), raw_site_chains_per_case=0,
        all30_L35_sparse_Qaction16_Ktarget2_Vtarget2_Znative_and_mergedaction_raw=True,
        native_only_all30_L35_target_start_to_UND_end_causal_suffix_KV_raw=True,
        sparse_raw_limit_bytes=MAX_BYTES, pre_forward_minimum_free_bytes=MIN_FREE_BYTES,
        actual_free_bytes_before_output_creation=available_before_stage, omitted_raw_live_SHA_not_independent_reconstruction=True,
        actual_packed_name_gate='Exactly two target IDs change; all other complete first kwargs/30 structural fields identical within scene/V.',
        background='Every masked site union66 all_allowed; selected current50 cut, action16 Z0, future200 and UND native.',
        historical_controls='Six X12 cream-cheese whole records and t0 actual boundaries equal completed language stage; new milk-box natives are first observations.',
        later_physics_preregistered=dict(trial_order=labels, cases=24, cached_q0=24, later_noise_base=195,
            q1_to_q7_seeds=list(range(196, 203)), later_predictions=168, later_model_forwards=5040,
            combined_model_forwards=5760, queries=192, later_whole_noise_pairs=161, physical_actions=3072,
            all24_execute_without_q0_ranking=True), physics_calls=0, no_training=True, no_residual_compensation=True, scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', **PRIMARY, scope=SCOPE), exclusive=True)
    write(out / 'sources.json', dict(source_evidence=context['source_evidence'], original_sources=context['original_noise_sources'],
        scenes=context['scenes'], frozen_contract=contract, canonical_references=context['canonical_references']), exclusive=True)
    started, completed, rows, runtime, observer, noise = time.perf_counter(), [], [], None, None, None
    counts = {key: 0 for key in expected_counts()}

    def progress(stage, **values):
        document = dict(stage=stage, completed=completed, expected_model_forwards=720, actual_counts=counts,
            elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', document)
        print('[MATCHED-NAMES] ' + json.dumps(document), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        language, value, frozen, factor = (context['modules'][key] for key in ('language', 'value', 'frozen', 'factor'))
        base, factory = load('matched_names_runtime', paths['normal_runtime']), load('matched_names_factory', paths['factory'])
        for goal in PROMPTS:
            base.TASKS[goal] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPTS[goal], TARGETS[goal])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        components, scan, reader, processor_module, dispatch, backend = value.runtime_contract(baseline, context, runtime)
        checker = components.CosmosComponentInterventions(model)
        originals, caches, references = {}, {}, {}
        for v in (195, 198):
            folder = Path(context['original_noise_sources'][str(v)]['q0_folder'])
            originals[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            caches[v] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(originals[v], torch)
            scan.validate_cache(caches[v], originals[v], checker, runtime, 'sealed original noise/' + str(v))
        runtime.require_exact(caches[195]['steps'], caches[198]['steps'], 'frozen schedule/domain/layout evidence')
        for label, reference in context['canonical_references'].items():
            references[label] = torch.load(Path(reference['folder']) / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
        prompt_ids = {goal: runtime.pipe.text_tokenizer.encode(prompt, add_special_tokens=False) for goal, prompt in PROMPTS.items()}
        for goal in PROMPTS:
            assert len(prompt_ids[goal]) == 11
            language.unique_span(prompt_ids[goal], TARGET_IDS[goal], 'actual tokenizer matched target/' + goal)
        native_random = runtime.cm.randn_tensor
        provenance = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), source_sha256={key: sha(path) for key, path in paths.items()},
            inherited_language_source_sha256=LANGUAGE_SHA, inherited_positions_source_sha256=POSITIONS_SHA,
            actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))),
            actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source_sha256=sha(Path(inspect.getfile(dispatch))), dispatch_signature=str(inspect.signature(dispatch)),
            torch_version=torch.__version__, CPU_threads=torch.get_num_threads(), actual_prompt_token_ids=prompt_ids,
            tokenizer_class=type(runtime.pipe.text_tokenizer).__name__, **backend, library_edits=False, physics_calls=0, scope=SCOPE)
        write(out / 'provenance.json', provenance, exclusive=True)
        Observer = observer_class(frozen, language)
        records, native_infos, scene_clean = {}, {}, {}
        for label, scene, goal, v, arm in PLAN:
            if arm != 'native':
                assert completed[:8] == labels[:8] and len(native_infos) == 8
            folder, image = out / label, Path(context['scenes'][scene]['image'])
            assert sha(image) == context['scenes'][scene]['files_sha256']['input.png']
            reference_label = f'cream_cheese_V{v}_A195_{arm}' if scene == 'x12' and goal == 'cream_cheese' else None
            reference = context['canonical_references'].get(reference_label)
            info = native_infos.get((scene, goal, v))
            observer = Observer(runtime, components, scan, reader, caches[195], None, originals[195], arm, folder,
                scene=scene, goal=goal, prompt_ids=prompt_ids[goal], native_info=info, legacy_reference=reference)
            noise = factor.InitialNoiseSources(runtime, scan, originals, v, 195)
            assert observer.original is dispatch and noise.original is native_random
            progress('fresh_q0', case=label)
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict(goal, image, 195, folder)
            finally:
                try:
                    noise.reset()
                finally:
                    observer.reset()
                    for key in counts:
                        counts[key] += observer.counts[key]
                    if folder.exists():
                        torch.save(dict(calls=noise.calls, model_steps=observer.calls, initial_model_kwargs=observer.first,
                            original_symbol_restored=noise.restored, observer_hooks_removed=not observer.handles,
                            dispatch_symbol_restored=observer.restored, actual_counts=observer.counts), folder / 'noise-audit.pt')
                        if observer.in_model:
                            torch.save(dict(step=observer.step, actual_model_kwargs=observer.actual_kwargs,
                                partial_raw_boundaries=observer.boundaries, partial_live_hidden_metadata=observer.hidden_metadata,
                                site_events=observer.events, actual_counts=observer.counts), folder / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(image) and metadata['prompt'] == PROMPTS[goal] and metadata['task_index'] == 7
            report, noise_report = observer.report(record), noise.report(record)
            preparation = input_checks(record, observer, originals, v, runtime)
            if reference is not None:
                runtime.require_exact(record, references[reference_label], 'ENTIRE accepted X12 canonical same-arm record')
            if scene in scene_clean:
                runtime.require_exact(observer.clean_current, scene_clean[scene], 'both goals/V actual own-scene conditioned frame')
            else:
                scene_clean[scene] = observer.clean_current
            info = dict(scene=scene, goal=goal, case=label, native_reference_case=f'{scene}_{goal}_V{v}_A195_native',
                actual_pack=observer.calls, first_model_kwargs=observer.first, actual_clean_current=observer.clean_current,
                original_solver_initial_action_FP32=record['action_states'][0], language=observer.language,
                actual_input_checks=preparation, scope=SCOPE)
            if arm == 'native':
                native_infos[(scene, goal, v)] = info
            else:
                native = records[info['native_reference_case']]
                runtime.require_exact(record['prepared_latents_and_masks'], native['prepared_latents_and_masks'], 'entire own scene/goal/V preparation')
                runtime.require_exact(record['pure_noise'], native['pure_noise'], 'entire own scene/goal/V returned noise')
            torch.save(info, folder / 'language-contract.pt')
            control = dict(state='exact', native_reference_case=info['native_reference_case'],
                own_scene_goal_V_initial_kwargs_and_all30_pack_exact=arm != 'native',
                existing_entire_canonical_record_exact=reference is not None, canonical_reference_case=reference_label,
                source_noise_preparation_sigma_clamp_padding_exact=True, independent_new_native_repeat_performed=False,
                no_X12_image_or_entire_vision_comparison_for_X15=True)
            write(folder / 'control-checks.json', control, exclusive=True)
            write(folder / 'observer-report.json', report, exclusive=True)
            write(folder / 'noise-audit.json', dict(noise=noise_report, preparation=preparation), exclusive=True)
            metadata.update(case=label, scene=scene, goal=goal, arm=arm, query=0, scene_task_index=7,
                policy_prompt=PROMPTS[goal], target_object=TARGETS[goal], shift_cm=SCENES[scene],
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                producer_script_sha256=provenance['script_sha256'], sparse_capture=True,
                full37_raw_steps=[0], full_QKV_site=dict(step=0, layer_zero_based=35), physical_prediction_evaluated=False, scope=SCOPE)
            write(folder / 'metadata.json', metadata)
            assert read(folder / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, scene=scene, goal=goal, prompt=PROMPTS[goal], arm=arm,
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                shift_cm=SCENES[scene], fresh_model_forwards=30, actual_dispatch_counts=observer.counts,
                existing_entire_canonical_record_exact=reference is not None, actual_und_len=122,
                actual_target_positions=observer.language['target_positions'], actual_target_token_ids=TARGET_IDS[goal],
                files_sha256={name: sha(folder / name) for name in FILES}, boundary_files=observer.files,
                dispatch_files=observer.dispatch_files,
                final_normalized_actions=record['actions'].tolist(), physical_prediction_evaluated=False))
            records[label] = record
            completed.append(label)
            if len(completed) == 8:
                write(out / 'matched-language-contract.json', matched_contract(native_infos, runtime), exclusive=True)
            assert disk_bytes(out) <= MAX_BYTES, 'Actual compact stage exceeds frozen 2.5GiB cap'
            progress('case_complete', case=label, actual_stage_bytes=disk_bytes(out))
            observer = noise = None
        for label, _, _, v, _ in PLAN:
            runtime.require_exact(records[label]['pure_noise'], originals[v]['pure_noise'][:1] + originals[195]['pure_noise'][1:], 'all24 whole selected V/A returned draws')
        assert counts == expected_counts() and sum(counts[key] for key in DISPATCH_KEYS) == 70200
        assert completed == labels and not checker._handles and checker._kind is None
        assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
        write(out / 'results.json', dict(state='complete', cases=rows, fresh_q0_predictions=24, fresh_model_forwards=720,
            official_dispatch_counts=counts, total_official_dispatch_calls=70200, primary_physical_prediction_evaluated=False,
            preregistered_physics_cases=labels, physics_calls=0, scope=SCOPE), exclusive=True)
        total = disk_bytes(out)
        assert total < MAX_BYTES - 1_048_576
        write(out / 'complete.json', dict(state='complete', script_sha256=sha(Path(__file__)),
            **{name.replace('-', '_') + '_sha256': sha(out / (name + '.json')) for name in
               ('results', 'provenance', 'protocol', 'sources', 'primary-prediction', 'matched-language-contract')},
            fresh_q0_predictions=24, fresh_model_forwards=720, official_dispatch_counts=counts,
            total_official_dispatch_calls=70200, original_dispatch_calls=51840, extra_dispatch_calls=18360,
            original_randn_calls_consumed=48, returned_source_draws_observed=48,
            six_X12_canonical_entire_accepted_language_records_exact=True,
            actual_equal_length_two_target_ID_packing_gate_passed=True,
            full37_UND_current_action_raw_steps=[0], full_model_kwargs_tuple_raw_steps=[0], full_QKV_raw_sites=24,
            all720_actual_L35_sparse_Qaction_target_KV_and_action_outputs_saved=True,
            all240_native_actual_L35_causal_suffix_KV_saved=True,
            all720_actual_final_action_input_output_timestep_sigma_raw_saved=True,
            all720x37_actual_hidden_arrays_finite_live_SHA_recorded=True,
            site_raw_chain_captures=0, actual_stage_bytes_before_completion=total, maximum_stage_bytes=MAX_BYTES,
            all_original_symbols_restored=True, all_owned_hooks_removed=True,
            primary_physical_prediction_evaluated=False, physics_calls=0, later_query_predictions=0, scope=SCOPE), exclusive=True)
        progress('complete')
        assert disk_bytes(out) <= MAX_BYTES
    except Exception as exc:
        write(out / 'failed.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc(), completed=completed,
            actual_counts=counts, active_case=None if observer is None else observer.folder.name,
            active_actual_counts=None if observer is None else observer.counts, physics_calls=0, scope=SCOPE))
        raise
    finally:
        try:
            if noise is not None:
                noise.reset()
        finally:
            if observer is not None:
                observer.reset()


if __name__ == '__main__':
    main()
