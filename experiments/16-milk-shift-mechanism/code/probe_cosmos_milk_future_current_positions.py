"""Twelve fresh q0 runs transferring the frozen first-nine-layer intervention.

--output BASE creates absent BASE/future-current-positions. CPU inputs are
collected separately under BASE/future-current-position-inputs/inputs.
X+15/18cm x V195/198, fixed A195 x native/all_allowed/early_l1_9:
360 model forwards, 35100 official dispatches, no physics or extra source
forwards. The first four native runs establish each new scene/V reference;
they are first observations, not independent repeat controls. Old X12 records
provide frozen random draws and structural/schedule evidence only.

source_contract(BASE) and prepared(BASE) return the same dictionary:
modules={groups,windows,frozen,factor}; frozen_contract; original_noise_sources;
source_evidence; paths; input_root; input_metadata; scenes; out. Modules and
Path objects are in-process interfaces. sources.json stores the serializable
evidence, noise sources, scene descriptors and frozen contract. PLAN entries
are (label, scene, vision_seed, arm), and label always includes scene.

All eight masked cases share the full36x30 union66 all-allowed background.
Only four early_l1_9 cases cut future200->current50 at steps0..14/layers0..8
(135 sites each). Native dispatch, UND/future returns, full266 projection,
unblocked-row exact gates and actual37 boundaries use frozen b5af/c106 code.
The sole subsequent physical criterion is four strictly only-milk cut cases
AND each numerical sham retaining its native classification. All twelve
128-step/8-query trials are preregistered; this q0 producer cannot evaluate
that criterion. Hard masks also redistribute attention. No training or
identity/function-region/root-cause claim.
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace


ROOT = Path('/home/current/work/cosmos3')
GROUP_SHA = 'b5af0c4c63dbeee37c81a783c8c7f1f73146a6c9c679c0130899756b7f835689'
EXECUTION_SHA = 'e6d5abcb19d866f372d1373ba97a383f8093f1917859ff9714451f8e6138d40f'
ANALYSIS_SHA = 'ca88ff9c7ee62beef9851fdfaea1c9fddd6e33c0f99058ebee16f26a28a4af9c'
INPUT_COLLECTOR_SHA = 'de6fb4a17640fd3c5def823d6a50c8f2a07aac72e9793b74b7ec15ca14f67cc1'
INPUT_DOCUMENT_SHA = {
    'complete.json': 'c73405d127b786643f27b6bba6b2930bbfd6720ede9ef2327b88a2baa750c0cf',
    'protocol.json': '1039110857f96bc9929b37c8d61a8eedbfd35a4e5e7f23667ff78f8780b982bb',
    'inputs/complete.json': 'fc64d190980ba41ca520cabca566d28a1606e5667fe762fb9d9d4be5e2ef139c',
    'inputs/metadata.json': '640aab50fef3d900877130058ab5aa538e5cf5f713fbce523455a28f66ef71ca',
    'inputs/x15/input.png': 'cb2b4354b9c4c32d0275fcf75ae45a8fb1dd22058303d39e2244d8397acc7788',
    'inputs/x18/input.png': '511b58bf7e6a3a27784d06e43320ceed74f5af749ab44055d169ba0065a0d60a'}
OLD_COMPLETE = {
    'future-current-layer-groups': '75731bc1b1cebf59b18181c3495f33cb194ad9ef0c44d2c45fbf26bb3c516d05',
    'future-current-layer-execution': '109d2e9fc014ab49cae4e70d7c2b7a22f90b2bbb8fa878c36696239b34b8dbe0',
    'future-current-layer-analysis': '75ae38a4ba03f958c93bee80b3a33f49a186f291eba35169e3c5b332270e6f48'}
PROMPT = 'pick up the milk and place it in the basket'
SCENES = {'x15': 15, 'x18': 18}
WINDOWS = {'early_l1_9': dict(steps=[0, 14], layers=[0, 8])}
ARMS = ('native', 'all_allowed', 'early_l1_9')
PLAN = tuple((f'{scene}_V{v}_A195_{arm}', scene, v, arm)
             for arm in ARMS for scene in SCENES for v in (195, 198))
PRIMARY_CASES = [label for label, _, _, arm in PLAN if arm == 'early_l1_9']
FILES = ('states.pt', 'normalized_actions.json', 'metadata.json', 'noise-audit.pt', 'noise-audit.json',
         'observer-report.json', 'indexes-and-masks.pt', 'control-checks.json', 'new-input-contract.pt')
SCOPE = ('Fixed first-nine-layer future-to-current cut transferred without tuning to X+15/18cm, '
         'two frozen V realizations with A195. Every native is a fresh new-scene reference; '
         'every sham/cut uses that same scene/V reference. Four cut cases and four numerical '
         'sham classification controls are preregistered for physics, without q0 selection. '
         'Only along-X transfer is tested. Hard masks also reallocate attention. '
         'No new physics, training, semantic anatomy or general root-cause claim.')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_contract(baseline):
    """Read-only preflight; no environment/model creation or torch import here."""
    group_path = ROOT / 'work/probe_cosmos_milk_future_current_layer_groups.py'
    execution_path = ROOT / 'work/run_cosmos_milk_future_current_layer_execution.py'
    analysis_path = ROOT / 'work/analyze_cosmos_milk_future_current_layer_groups.py'
    assert sha(group_path) == GROUP_SHA and sha(execution_path) == EXECUTION_SHA and sha(analysis_path) == ANALYSIS_SHA
    groups = load('positions_frozen_layer_groups', group_path)
    windows, frozen, factor, contract, noise_sources, _, paths, earlier = groups.source_contract(baseline)
    assert groups.WINDOWS['early_l1_9'] == WINDOWS['early_l1_9']
    stages, documents = {}, {}
    for stage, expected in OLD_COMPLETE.items():
        folder = baseline / stage
        assert not any((folder / name).exists() for name in ('failed.json', 'wrapper_failed.json', 'server/failed.json', 'closed-loop/failed.json'))
        assert sha(folder / 'complete.json') == expected
        complete = read(folder / 'complete.json')
        assert complete['state'] == 'complete'
        hashes = dict(complete=expected)
        if stage.endswith('groups'):
            assert complete['script_sha256'] == GROUP_SHA and complete['fresh_model_forwards'] == 360
            assert complete['total_official_dispatch_calls'] == 37800
            for name in ('results', 'provenance', 'protocol', 'sources', 'primary-prediction'):
                key = 'primary_prediction_sha256' if name == 'primary-prediction' else name + '_sha256'
                assert sha(folder / (name + '.json')) == complete[key]
                hashes[name] = complete[key]
        else:
            assert complete['script_sha256'] == (EXECUTION_SHA if stage.endswith('execution') else ANALYSIS_SHA)
            for name, expected_file in complete['files_sha256'].items():
                assert sha(folder / name) == expected_file
                hashes[name] = expected_file
        stages[stage], documents[stage] = complete, hashes
    producer = baseline / 'future-current-layer-groups'
    results, sources = read(producer / 'results.json'), read(producer / 'sources.json')
    assert sources == dict(source_evidence=earlier, original_sources=noise_sources, frozen_contract=contract)
    assert results['official_dispatch_counts'] == dict(model_before=360, model_after=360, native_UND_dispatch=12960,
        native_GEN_dispatch=12960, all_allowed_dispatch=10800, cut_dispatch=1080)
    assert [row['case'] for row in results['cases']] == [row[0] for row in groups.PLAN]
    q0 = {}
    for row, (label, v, arm) in zip(results['cases'], groups.PLAN):
        folder = producer / label
        assert row['arm'] == arm and row['vision_noise_source_seed'] == v and row['fresh_model_forwards'] == 30
        assert row['action_noise_source_seed'] == row['runtime_rng_seed'] == 195
        for name, expected in row['files_sha256'].items():
            assert sha(folder / name) == expected
        assert len(row['boundary_files']) == 30
        for step, item in enumerate(row['boundary_files']):
            assert item['step'] == step and item['file'] == f'boundaries-t{step:02d}.pt'
            assert sha(folder / item['file']) == item['sha256']
        q0[label] = dict(folder=str(folder), files_sha256=row['files_sha256'], boundary_files=row['boundary_files'])
    physics = baseline / 'future-current-layer-execution'
    pc, controls, primary, pprov = (read(physics / (name + '.json')) for name in ('complete', 'controls', 'primary-prediction', 'provenance'))
    assert pc['physical_trials'] == 12 and pc['fresh_model_forwards'] == 2520 and pc['actual_later_whole_noise_pairs'] == 77
    assert pc['primary_prediction'] == primary['prediction'] == 'supported' and pc['primary_both_only_milk'] is True
    assert pc['numerical_sham_classification_gate_passed'] is True and primary['both_only_milk'] is True
    assert pprov['producer_documents_sha256'] == {name: sha(producer / (name + '.json')) for name in
        ('complete', 'results', 'provenance', 'protocol', 'sources', 'primary-prediction')}
    assert pprov['source_evidence'] == earlier
    for label, hashes in controls['actual_case_files_sha256'].items():
        for name, expected in hashes.items():
            assert sha(physics / 'closed-loop' / label / name) == expected
    ac = stages['future-current-layer-analysis']
    assert ac['primary_prediction'] == 'supported' and ac['actual_physical_actions_checked'] == 1536
    assert ac['whole_noise_pairs_checked'] == 77 and ac['whole_model_records_checked'] == 96
    assert ac['actual_boundary_files_checked'] == 360 and ac['active_cut_sites_checked'] == 1080
    assert ac['q0_total_official_dispatch_calls'] == 37800 and ac['cuda_initialized'] is False
    assert ac['model_forwards'] == ac['attention_dispatch_calls'] == ac['solver_steps'] == ac['physics_calls'] == 0
    aprov = read(baseline / 'future-current-layer-analysis/provenance.json')
    assert aprov['producer_script_sha256'] == GROUP_SHA and aprov['execution_script_sha256'] == EXECUTION_SHA
    assert aprov['source_execution_complete_sha256'] == OLD_COMPLETE['future-current-layer-execution']
    evidence = dict(frozen_X12_evidence=earlier, completed_nine_layer_stages=documents,
        completed_nine_layer_q0=q0, prior_primary='supported', old_eighteen_layer_primary='rejected',
        source_scope='X12 clean images/latents/full records are NOT new-position references. Only sealed raw FP32 draws, code and schedule/layout evidence are reused.')
    input_root = baseline / 'future-current-position-inputs/inputs'
    for name, expected in INPUT_DOCUMENT_SHA.items():
        assert sha(input_root.parent / name) == expected
    collector_wrapper = ROOT / 'work/collect_cosmos_milk_future_current_positions.py'
    assert sha(collector_wrapper) == INPUT_COLLECTOR_SHA
    outer, outer_protocol = read(input_root.parent / 'complete.json'), read(input_root.parent / 'protocol.json')
    assert not (input_root.parent / 'failed.json').exists()
    assert outer['state'] == 'complete' and outer['script_sha256'] == outer_protocol['script_sha256'] == INPUT_COLLECTOR_SHA
    assert outer['protocol_sha256'] == sha(input_root.parent / 'protocol.json')
    assert outer['inputs_complete_sha256'] == sha(input_root / 'complete.json')
    assert outer['inputs_metadata_sha256'] == sha(input_root / 'metadata.json')
    assert outer['policy_model_forwards'] == outer_protocol['policy_model_forwards'] == 0
    assert outer['policy_scenes'] == outer_protocol['policy_scenes'] == list(SCENES)
    assert outer['reference_only_scene'] == outer_protocol['reference_only_scene'] == 'x00'
    assert outer['additional_warmup_steps'] == 0 and outer['all_three_scenes_typed_state_controller_pixels_restored_exact'] is True
    assert outer_protocol['shifts_cm'] == {'x00': 0, 'x15': 15, 'x18': 18} and outer_protocol['prompt'] == PROMPT
    assert outer_protocol['no_env_step_during_collection_or_restore'] is True
    assert outer_protocol['fixed_followup'] == dict(vision_source_seeds=[195, 198], action_source_seed=195,
        arms=list(ARMS), fresh_q0_predictions=12, selected_window=WINDOWS['early_l1_9'], no_selection_by_outcome=True)
    for role, frozen_role in (('simulator', 'simulator'), ('collector', 'collector'), ('controller', 'controller_helper')):
        assert outer_protocol['source_files'][role] == contract['frozen_files'][frozen_role]
    ic, im = read(input_root / 'complete.json'), read(input_root / 'metadata.json')
    assert ic['state'] == 'matched_inputs_verified' and ic['metadata_sha256'] == sha(input_root / 'metadata.json')
    assert ic['additional_warmup_steps'] == 0 and ic['policy_executed'] is False and ic['source_center_pixels_exact'] is True
    assert not (input_root / 'failed.json').exists()
    assert im['shifts_cm'] == {'x00': 0, 'x15': 15, 'x18': 18} and im['prompt'] == PROMPT
    assert im['additional_warmup_steps'] == 0 and im['source_is_already_warmed'] is True
    assert im['suite'] == 'libero_object' and im['task_index'] == 7 and im['task_order_index'] == 0
    for key in ('source_state', 'source_png'):
        assert sha(Path(im[key])) == im[key + '_sha256']
        assert dict(path=im[key], sha256=im[key + '_sha256']) == contract['frozen_files'][key]
    helper_path = Path(contract['frozen_files']['simulator']['path'])
    collector_path = Path(contract['frozen_files']['collector']['path'])
    controller_path = Path(contract['frozen_files']['controller_helper']['path'])
    for key, path in (('script', helper_path), ('collector', collector_path), ('controller_helper', controller_path)):
        assert sha(path) == im[key + '_sha256']
    simulator, collector, controller = (load('position_inputs_' + key, path) for key, path in
        (('milk_capture', helper_path), ('collector', collector_path), ('controller', controller_path)))
    scenes = {}
    for scene, shift in SCENES.items():
        folder, description = input_root / scene, im['scenes'][scene]
        assert description['shift_cm'] == shift and description['changed_state_indexes'] == [1 + description['milk_qpos_address']]
        for field, name in (('input_png_sha256', 'input.png'), ('state_sha256', 'state.npy'),
                            ('arrays_sha256', 'arrays.json'), ('controller_sha256', 'controller.npz')):
            assert sha(folder / name) == description[field]
        actual = simulator.load_capture(folder, im, collector)
        assert actual['state']['sim_state'].dtype == simulator.np.float64
        assert all(item['exact'] for name in ('restore_checks', 'center_restore_checks') for item in read(folder / (name + '.json')))
        for name in ('controller_checks', 'restored_controller_checks'):
            assert all(read(folder / (name + '.json')).values())
        contact = read(folder / 'initial_contact_checks.json')
        assert contact['interobject_pairs'] == contact['milk_robot_contacts'] == []
        key = 'milk_1__' + description['milk_joint'] + '__qpos'
        assert controller.array_exact(actual['objects'][key], simulator.np.asarray(description['milk_qpos']))
        with simulator.np.load(folder / 'controller.npz', allow_pickle=False) as archive:
            assert len(archive.files) == 10 and all(simulator.np.isfinite(archive[name]).all() for name in archive.files)
        names = ('state.npy', 'arrays.json', 'fixtures.npz', 'cameras.npz', 'observations.npz', 'objects.npz', 'controller.npz',
                 'input.png', 'agentview.png', 'robot0_eye_in_hand.png', 'restore_checks.json', 'center_restore_checks.json',
                 'controller_checks.json', 'restored_controller_checks.json', 'initial_contact_checks.json')
        scenes[scene] = dict(folder=str(folder), image=str(folder / 'input.png'), shift_cm=shift,
            files_sha256={name: sha(folder / name) for name in names}, physical_scene=description,
            hash_scope='New CPU restored real scene and observations; no X12 image/input symlink.')
    evidence['new_position_inputs'] = dict(root=str(input_root), metadata_sha256=sha(input_root / 'metadata.json'),
        complete_sha256=sha(input_root / 'complete.json'), scenes=scenes,
        collector_wrapper_sha256=INPUT_COLLECTOR_SHA, collector_complete_sha256=sha(input_root.parent / 'complete.json'),
        collector_protocol_sha256=sha(input_root.parent / 'protocol.json'))
    return dict(modules=dict(groups=groups, windows=windows, frozen=frozen, factor=factor), frozen_contract=contract,
        original_noise_sources=noise_sources, source_evidence=evidence, paths=paths, input_root=input_root,
        input_metadata=im, scenes=scenes, out=baseline / 'future-current-positions')


def prepared(baseline):
    """Same dict as source_contract; consumer performs its own absent-output preparation."""
    return source_contract(baseline)


def observer_class(groups, windows, frozen):
    Parent = groups.observer_class(windows, frozen)

    class PositionObserver(Parent):
        def __init__(self, *args, case_arm, scene, **kwargs):
            super().__init__(*args, case_arm=case_arm, **kwargs)
            self.scene = scene
            self.window = WINDOWS.get(case_arm)

        def before_model(self, module, args, kwargs):
            if self.case_arm != 'native':
                super().before_model(module, args, kwargs)
                return
            # Bootstrap must NOT call the parent's X12 full-input comparison.
            assert not args and not self.in_model and self.active is None
            self.step += 1
            assert self.step < 30 and kwargs['return_dict'] is False
            layout = self.components._layout(kwargs)
            actual = dict(layout=layout, metadata={key: self.runtime.cpu(kwargs.get(key)) for key in self.components._STRUCTURAL_KEYS})
            self.runtime.require_exact(actual, self.cache['steps'][self.step], 'frozen structural/schedule only/' + str(self.step))
            derived = frozen.indexes(kwargs, self.model, self.components, self.runtime)
            if self.index is None:
                self.index = derived
                self.masks = frozen.masks_for(derived, self.torch)
                self.gpu_masks = {name: item['mask'].to(kwargs['vision_tokens'][0].device) for name, item in self.masks.items()}
                self.first = self.runtime.cpu(kwargs)
                self.clean_current = self.first['vision_tokens'][0][0, :, 0].clone()
                self.torch.save(dict(indexes=self.index, masks=self.masks), self.folder / 'indexes-and-masks.pt')
            else:
                self.runtime.require_exact(derived, self.index, 'all-step actual new-scene index mapping')
            self.runtime.require_exact(self.runtime.cpu(kwargs['vision_tokens'][0][0, :, 0]), self.clean_current, 'new-scene current clamp/' + str(self.step))
            assert self.torch.count_nonzero(kwargs['action_tokens'][0][:, 10:]) == 0
            self.calls.append(actual)
            self.counts['model_before'] += 1
            self.in_model, self.next_layer = True, 0
            self.boundaries, self.events = [], []
            self.actual_kwargs = self.runtime.cpu(kwargs)

        def report(self, record):
            value = super().report(record)
            value.update(scene=self.scene, scope=SCOPE, native_source_first_observation=self.case_arm == 'native',
                independent_nohook_repeat_performed=False, X12_entire_input_comparison_performed=False)
            return value
    return PositionObserver


def actual_input_checks(record, observer, source_noise, runtime, folder):
    """Independent inner record vs actual outer kwargs and FP32 solver states."""
    torch = runtime.torch
    runtime.require_exact(record['model_input'], observer.first, 'inner/outer first model kwargs')
    for key in ('timesteps', 'sigmas'):
        runtime.require_exact(record[key], source_noise[key], 'official full schedule/' + key)
    assert record['action_states'].shape == (31, 1, 16, 64) and record['action_states'].dtype == torch.float32
    assert torch.count_nonzero(record['action_states'][..., 10:]) == 0
    prepared = record['prepared_latents_and_masks']
    assert type(prepared) is tuple and len(prepared) == 12
    runtime.require_exact(prepared[0][0, :, 0].to(torch.bfloat16), observer.clean_current, 'actual new-scene FP32 prepared current to BF16 kwargs')
    runtime.require_exact(prepared[2], source_noise['prepared_latents_and_masks'][2], 'A195 FP32 initial preparation')
    runtime.require_exact(record['action_states'][0], source_noise['action_states'][0], 'A195 FP32 solver initial sample')
    for step, item in enumerate(observer.files):
        boundary = torch.load(folder / item['file'], map_location='cpu', weights_only=True, mmap=True)
        kwargs = boundary['actual_model_kwargs']
        runtime.require_exact(kwargs['action_tokens'][0], record['action_states'][step][0].to(torch.bfloat16), 'FP32 solver sample to actual BF16 input/' + str(step))
        runtime.require_exact(kwargs['vision_tokens'][0][0, :, 0], observer.clean_current, 'actual current condition/' + str(step))
        runtime.require_exact(boundary['model_output'][2][0], record['action_velocity'][step], 'actual tuple action head/' + str(step))
        runtime.require_exact(boundary['solver_sigma_from_same_frozen_schedule'], record['sigmas'][step], 'actual saved sigma/' + str(step))
    return dict(inner_outer_first_kwargs_byte_exact=True, all30_actual_FP32_solver_sample_to_BF16_action_kwargs_exact=True,
        new_scene_prepared_current_FP32_cast_to_actual_BF16_frame0_exact=True, all30_current_condition_frame0_exact=True,
        official_schedule_exact=True, all31_solver_padding54_zero=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed BASE plus separate CPU future-current-position-inputs/inputs')
    baseline = parser.parse_args().output.resolve()
    context = source_contract(baseline)
    groups, windows, frozen, factor = (context['modules'][name] for name in ('groups', 'windows', 'frozen', 'factor'))
    contract, paths, sources, out = (context[name] for name in ('frozen_contract', 'paths', 'source_evidence', 'out'))
    out.mkdir(exist_ok=False)
    labels = [item[0] for item in PLAN]
    protocol = dict(plan=[dict(case=label, scene=scene, shift_cm=SCENES[scene], vision_noise_source_seed=v,
        action_noise_source_seed=195, runtime_rng_seed=195, arm=arm, window=WINDOWS.get(arm)) for label, scene, v, arm in PLAN],
        trial_order=labels, case_scene={label: scene for label, scene, _, _ in PLAN}, scenes=context['scenes'], prompt=PROMPT,
        windows_zero_based_inclusive=WINDOWS, primary_window='early_l1_9', primary_cases=PRIMARY_CASES,
        primary_evaluation=dict(status='not_evaluated_q0_only', all_four_cut_selected_objects_must_equal=['milk_1'],
            numerical_shams_must_preserve_own_native_selected_objects=True, no_other_location_or_window_promoted=True),
        new_native_reference_rule='First four native q0 are included in twelve, with actual new-scene full kwargs/preparation/records. No old X12 current/image/full-record substitute or independent nohook repeat.',
        original_randn_rule='Consume true A195 RNG draw, then return sealed V/A full FP32 source draw; two actual draws per prediction.',
        source_scene_full_input_rule='Same scene AND V fresh native is the initial full-kwargs reference for sham/cut.',
        future_current_cut='Only future200 keys to current50 queries; action-to-current retained.',
        masked_sites_per_case=1080, cut_sites_per_window_case=135, common_masked_union66=True,
        actual_model_forwards=360, total_official_dispatch_calls=35100, original_dispatch_calls=25920,
        all_allowed_dispatch_calls=8640, cut_dispatch_calls=540, extra_dispatch_calls=9180,
        original_randn_calls_consumed=24, no_extra_source_forwards=True, physics_calls=0,
        later_physics_preregistered=dict(trial_order=labels, cases=12, cached_q0=12, later_noise_base=195,
            q1_to_q7_seeds=list(range(196, 203)), requests=96, later_predictions=84, later_model_forwards=2520,
            total_two_stage_model_forwards=2880, steps_per_case=128, queries_per_case=8, whole_noise_pairs_expected=77,
            all12_execute_without_q0_score_selection=True, stop_native_interpretation_if_sham_changes_classification=True,
            execute_all12_even_if_sham_changes_classification=True), scope=SCOPE)
    write(out / 'protocol.json', protocol, exclusive=True)
    write(out / 'sources.json', dict(source_evidence=sources, original_noise_sources=context['original_noise_sources'],
        scenes=context['scenes'], frozen_contract=contract), exclusive=True)
    write(out / 'primary-prediction.json', dict(status='not_evaluated_q0_only', cases=PRIMARY_CASES,
        all_four_selected_objects_must_equal=['milk_1'], sham_preserves_own_native_classification_required=True,
        all12_physical_cases_preregistered=True, q0_numeric_rank_used=False, scope=SCOPE), exclusive=True)
    started, completed, rows, runtime, observer, noise = time.perf_counter(), [], [], None, None, None
    counts = dict(model_before=0, model_after=0, native_UND_dispatch=0, native_GEN_dispatch=0, all_allowed_dispatch=0, cut_dispatch=0)

    def progress(stage, **values):
        value = dict(stage=stage, completed=completed, q0_predictions_complete=len(completed),
            completed_model_forwards=30 * len(completed), actual_counts=counts, elapsed_s=time.perf_counter() - started, **values)
        write(out / 'progress.json', value)
        print('[FUTURE-CURRENT-POSITIONS] ' + json.dumps(value), flush=True)

    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        base = load('positions_normal_runtime', paths['normal_runtime'])
        base.TASKS['milk'] = (7, 'pick_up_the_milk_and_place_it_in_the_basket', PROMPT, 'milk_1')
        factory = load('positions_factory', paths['factory'])
        factory.OUT = out
        assert sha(factory.SCHEDULER) == contract['frozen_files']['scheduler']['sha256']
        progress('loading_model')
        runtime = base.NormalRuntime(factory)
        torch, model = runtime.torch, runtime.pipe.transformer
        assert not model.training and not model.is_cache_enabled and not torch.is_grad_enabled()
        upstream = sources['frozen_X12_evidence']['anchors']['upstream']
        assert sha(Path(inspect.getfile(type(model)))) == upstream['actual_transformer_sha256']
        assert sha(Path(inspect.getfile(type(runtime.pipe)))) == upstream['actual_pipeline_source_sha256']
        components, scan, reader = (load('positions_' + key, paths[key]) for key in ('components', 'metrics', 'reader'))
        checker = components.CosmosComponentInterventions(model)
        processor_module = inspect.getmodule(type(model.layers[0].self_attn.processor))
        dispatch = processor_module.dispatch_attention_fn
        dispatch_path = Path(inspect.getfile(dispatch))
        assert sha(dispatch_path) == frozen.DISPATCH_SHA
        assert hashlib.sha256(inspect.getsource(dispatch).encode()).hexdigest() == read(baseline / 'target-reader-inputs/provenance.json')['dispatch_function_source_sha256']
        dispatch_module = importlib.import_module('diffusers.models.attention_dispatch')
        assert dispatch_module.dispatch_attention_fn is dispatch
        registry = dispatch_module._AttentionBackendRegistry
        backend_name, backend_fn = registry.get_active_backend()
        supported = registry._supported_arg_names[backend_name]
        feasibility = sources['frozen_X12_evidence']['anchors']['feasibility_actual_registry']
        assert 'attn_mask' in supported and backend_name.value != '_native_flash'
        assert backend_name.value == feasibility['registry_backend'] and backend_fn.__qualname__ == feasibility['registry_backend_function']
        assert hashlib.sha256(inspect.getsource(backend_fn).encode()).hexdigest() == feasibility['registry_backend_function_source_sha256']
        assert sorted(supported) == feasibility['registry_supported_arguments']
        settings = reader.settings(torch, model.layers[0].self_attn.processor)
        for key, value in feasibility['settings'].items():
            assert settings[key] == value
        for block in model.layers:
            assert block.self_attn.processor._parallel_config is block.self_attn.processor._attention_backend is None
            assert reader.settings(torch, block.self_attn.processor) == settings
        raw_sources, old_caches = {}, {}
        for v in (195, 198):
            folder = Path(context['original_noise_sources'][str(v)]['q0_folder'])
            raw_sources[v] = torch.load(folder / 'states.pt', map_location='cpu', weights_only=True, mmap=True)
            old_caches[v] = torch.load(folder / 'components.pt', map_location='cpu', weights_only=True, mmap=True)
            scan.validate_record(raw_sources[v], torch)
            scan.validate_cache(old_caches[v], raw_sources[v], checker, runtime, 'old-frozen-structural/' + str(v))
            assert len(raw_sources[v]['pure_noise']) == 2 and torch.count_nonzero(raw_sources[v]['action_states'][..., 10:]) == 0
        runtime.require_exact(old_caches[195]['steps'], old_caches[198]['steps'], 'frozen complete schedule and pack metadata only')
        native_random = runtime.cm.randn_tensor
        write(out / 'provenance.json', dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(out / 'protocol.json'),
            sources_sha256=sha(out / 'sources.json'), source_sha256={key: sha(path) for key, path in paths.items()},
            inherited_first9_source_sha256=GROUP_SHA, inherited_window_source_sha256=groups.WINDOW_PROBE_SHA,
            inherited_dispatcher_source_sha256=groups.BASE_PROBE_SHA, noise_seam_source_sha256=frozen.FACTOR_SHA,
            actual_transformer_sha256=sha(Path(inspect.getfile(type(model)))), actual_pipeline_source_sha256=sha(Path(inspect.getfile(type(runtime.pipe)))),
            dispatch_source_sha256=sha(dispatch_path), dispatch_signature=str(inspect.signature(dispatch)),
            registry_backend=backend_name.value, backend_settings=settings, source_evidence=sources,
            new_source_first_observations=True, extra_source_forwards=0, independent_nohook_repeat_performed=False,
            library_edits=False, training=False, physics_calls=0, scope=SCOPE), exclusive=True)
        Observer = observer_class(groups, windows, frozen)
        records, fresh_sources, fresh_caches, clean_currents, native_observers = {}, {s: {} for s in SCENES}, {}, {}, {}
        for label, scene, v, arm in PLAN:
            if arm != 'native':
                assert completed[:4] == labels[:4] and all(len(fresh_sources[s]) == 2 for s in SCENES)
            directory, image = out / label, Path(context['scenes'][scene]['image'])
            assert sha(image) == context['scenes'][scene]['files_sha256']['input.png']
            progress('fresh_q0', case=label, scene=scene)
            source = raw_sources[v] if arm == 'native' else fresh_sources[scene][v]
            cache = old_caches[195] if arm == 'native' else fresh_caches[(scene, v)]
            clean = None if arm == 'native' else clean_currents[scene]
            observer = Observer(runtime, components, scan, reader, cache, clean, source,
                'native' if arm == 'native' else 'all_allowed', directory, case_arm=arm, scene=scene)
            noise = factor.InitialNoiseSources(runtime, scan, raw_sources, v, 195)
            assert observer.original is dispatch and noise.original is native_random
            observer.begin()
            try:
                noise.begin()
                record, metadata = runtime.predict('milk', image, 195, directory)
            finally:
                try:
                    noise.reset()
                finally:
                    observer.reset()
                    for key in counts:
                        counts[key] += observer.counts[key]
                    if directory.exists():
                        torch.save(dict(calls=noise.calls, model_steps=observer.calls, initial_model_kwargs=observer.first,
                            original_symbol_restored=noise.restored, observer_hooks_removed=not observer.handles,
                            dispatch_symbol_restored=observer.restored, actual_counts=observer.counts), directory / 'noise-audit.pt')
                        if observer.in_model:
                            torch.save(dict(step=observer.step, actual_model_kwargs=observer.actual_kwargs,
                                partial_current_action_boundaries=observer.boundaries, site_events=observer.events,
                                actual_counts=observer.counts), directory / 'partial-forward.pt')
            assert runtime.cm.randn_tensor is native_random and processor_module.dispatch_attention_fn is dispatch
            scan.validate_record(record, torch)
            assert metadata['model_calls'] == 30 and metadata['prepare_calls'] == 1 and metadata['seed'] == 195
            assert metadata['input_png_sha256'] == sha(image) and metadata['prompt'] == PROMPT and metadata['task_index'] == 7
            report, noise_report = observer.report(record), noise.report(record)
            actual = actual_input_checks(record, observer, raw_sources[195], runtime, directory)
            runtime.require_exact(record['prepared_latents_and_masks'][0][:, :, 1:], raw_sources[v]['prepared_latents_and_masks'][0][:, :, 1:], 'frozen V-source future preparation only')
            if arm == 'native':
                fresh_sources[scene][v] = record
                fresh_caches[(scene, v)] = dict(steps=observer.calls, first_model_kwargs=observer.first)
                native_observers[(scene, v)] = SimpleNamespace(first=observer.first, calls=observer.calls, handle=None)
                if scene in clean_currents:
                    runtime.require_exact(observer.clean_current, clean_currents[scene], 'both V sources actual new-scene current condition')
                else:
                    clean_currents[scene] = observer.clean_current
                control = dict(state='new_native_reference_observed', scene=scene, vision_noise_source_seed=v,
                    whole_prior_record_exact=None, independent_nohook_repeat_performed=False, actual_input_checks=actual)
                if len(fresh_sources[scene]) == 2:
                    checks = {}
                    for source_v in (195, 198):
                        native_label = f'{scene}_V{source_v}_A195_native'
                        checks[str(source_v)] = factor.prepared_contract(fresh_sources[scene][source_v], fresh_sources[scene], source_v, 195,
                            runtime, native_observers[(scene, source_v)])
                    write(out / (scene + '-new-native-source-checks.json'), checks, exclusive=True)
            else:
                prepared_report = factor.prepared_contract(record, fresh_sources[scene], v, 195, runtime,
                    SimpleNamespace(first=observer.first, calls=observer.calls, handle=None))
                if arm in WINDOWS:
                    sham = f'{scene}_V{v}_A195_all_allowed'
                    control = windows.prefix_checks(directory, out / sham, record, records[sham], WINDOWS[arm], runtime)
                    control.update(scene=scene, comparison_case=sham, comparison_same_scene_and_V=True)
                else:
                    control = dict(state='new_numerical_sham_observed', scene=scene, whole_prior_record_exact=None,
                        independent_nohook_repeat_performed=False, prepared_initial_full_input_equals_own_new_native=True)
                control['actual_input_checks'] = actual
                control['preparation'] = prepared_report
            for key in ('pure_noise', 'timesteps', 'sigmas'):
                runtime.require_exact(record[key], fresh_sources[scene][v][key], 'same-scene/V native noise and schedule/' + key)
            records[label] = record
            torch.save(dict(scene=scene, vision_noise_source_seed=v, native_reference_case=f'{scene}_V{v}_A195_native',
                actual_pack=observer.calls, first_model_kwargs=observer.first, actual_clean_current=observer.clean_current,
                original_solver_initial_action_FP32=record['action_states'][0], actual_input_checks=actual,
                reference_scope='Actual new-scene inputs; no X12 image, condition or full-model-input reference.'), directory / 'new-input-contract.pt')
            write(directory / 'control-checks.json', control, exclusive=True)
            write(directory / 'observer-report.json', report, exclusive=True)
            write(directory / 'noise-audit.json', dict(noise=noise_report, actual_input_checks=actual), exclusive=True)
            metadata.update(case=label, scene=scene, arm=arm, window_zero_based_inclusive=WINDOWS.get(arm), query=0, shift_cm=SCENES[scene],
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195,
                numerical_masked_sham=arm == 'all_allowed', native_noop=arm == 'native',
                common_union66_all_allowed_background=arm != 'native', window_cut_sites=observer.counts['cut_dispatch'],
                actual_masked_site_substitutions=observer.counts['substituted_sites'], primary_prediction_case=arm == 'early_l1_9',
                physical_prediction_evaluated=False, prior_entire_record_exact=None, new_native_reference_first_observation=arm == 'native',
                own_sham_pre_first_cut_prefix_exact=True if arm in WINDOWS else None)
            write(directory / 'metadata.json', metadata)
            assert read(directory / 'normalized_actions.json') == record['actions'].tolist()
            rows.append(dict(case=label, scene=scene, shift_cm=SCENES[scene], arm=arm, window=WINDOWS.get(arm),
                vision_noise_source_seed=v, action_noise_source_seed=195, runtime_rng_seed=195, fresh_model_forwards=30,
                actual_dispatch_counts=observer.counts, boundary_files=observer.files,
                files_sha256={name: sha(directory / name) for name in FILES},
                new_native_reference_first_observation=arm == 'native', physical_prediction_evaluated=False,
                own_sham_pre_first_cut_prefix_exact=True if arm in WINDOWS else None,
                final_normalized_actions=record['actions'].tolist()))
            completed.append(label)
            progress('case_complete', case=label, scene=scene)
        assert completed == labels and counts == dict(model_before=360, model_after=360, native_UND_dispatch=12960,
            native_GEN_dispatch=12960, all_allowed_dispatch=8640, cut_dispatch=540)
        for scene in SCENES:
            for arm in ARMS:
                a, b = (records[f'{scene}_V{v}_A195_{arm}'] for v in (195, 198))
                runtime.require_exact(a['pure_noise'][1], b['pure_noise'][1], 'cross-V fixed A195 actual draw')
                assert not frozen.byte_exact(a['pure_noise'][0], b['pure_noise'][0], torch)
        for v in (195, 198):
            a, b = (records[f'{scene}_V{v}_A195_native'] for scene in SCENES)
            runtime.require_exact(a['pure_noise'], b['pure_noise'], 'cross-position same V/A actual full draws')
        result = dict(state='complete', cases=rows, fresh_q0_predictions=12, fresh_model_forwards=360,
            total_official_dispatch_calls=35100, official_dispatch_counts=counts,
            new_native_source_checks_sha256={scene: sha(out / (scene + '-new-native-source-checks.json')) for scene in SCENES},
            primary_prediction_sha256=sha(out / 'primary-prediction.json'), preregistered_physics_cases=labels,
            primary_physical_prediction_evaluated=False, physics_calls=0, later_query_predictions=0, scope=SCOPE)
        write(out / 'results.json', result, exclusive=True)
        complete = dict(state='complete', script_sha256=sha(Path(__file__)), fresh_q0_predictions=12, fresh_model_forwards=360,
            total_official_dispatch_calls=35100, original_dispatch_calls=25920, extra_dispatch_calls=9180,
            original_randn_calls_consumed=24, returned_source_draws_observed=24, new_native_reference_predictions=4,
            no_extra_source_forwards=True, four_new_native_actual_scene_input_contracts_passed=True,
            all8_masked_initial_inputs_equal_own_scene_V_native=True, all4_cut_before_first_cut_own_scene_V_sham_prefix_exact=True,
            all_active_cut_unblocked_GEN_vs_same_input_allallowed_exact=True, all8_masked_full1080_sites_return_common_union66=True,
            all360_actual_complete_model_kwargs_saved=True, all_original_symbols_restored=True, all_owned_hooks_removed=True,
            primary_physical_prediction_evaluated=False, physics_calls=0, later_query_predictions=0,
            results_sha256=sha(out / 'results.json'), protocol_sha256=sha(out / 'protocol.json'), sources_sha256=sha(out / 'sources.json'),
            provenance_sha256=sha(out / 'provenance.json'), primary_prediction_sha256=sha(out / 'primary-prediction.json'),
            new_native_source_checks_sha256=result['new_native_source_checks_sha256'], scope=SCOPE)
        write(out / 'complete.json', complete, exclusive=True)
        progress('complete')
    except Exception as exc:
        write(out / 'failed.json', dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            completed=completed, actual_counts=counts, elapsed_s=time.perf_counter() - started))
        raise
    finally:
        try:
            if noise is not None and noise.installed:
                noise.reset()
        finally:
            if observer is not None:
                observer.reset()


if __name__ == '__main__':
    main()
