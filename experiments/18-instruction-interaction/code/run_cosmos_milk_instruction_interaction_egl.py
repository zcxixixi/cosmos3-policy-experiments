"""Execute the unchanged eight cases after an EGL-only launch correction.

The original attempt failed before environment creation, predictions or control
steps. Preserve it. Reuse the eight completed q0 records without recomputation.
The frozen execution body, all scientific cases and byte gates remain unchanged.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path


BODY_SHA = '1e9e5ab03df34d333ee0905529af2de53c3a872da6139fa65cefa0aac2b55876'
FAILED_SHA = '4af7ac7aa6c9d80a9e1c9061073b3c5f4e94ad6cdd9cf9db67f6d57b062e3c2e'
STAGE = 'future-instruction-interaction-execution-egl'
BODY = Path('/home/current/work/cosmos3/work/run_cosmos_milk_instruction_interaction.py')


def main():
    assert os.environ.get('MUJOCO_GL') == 'egl'
    assert os.environ.get('MUJOCO_EGL_DEVICE_ID') == '0'
    assert hashlib.sha256(BODY.read_bytes()).hexdigest() == BODY_SHA
    spec = importlib.util.spec_from_file_location('instruction_egl_body', BODY)
    body = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(body)
    assert body.STAGE == 'future-instruction-interaction-execution'
    original_contract = body.contract

    def contract(baseline):
        failed = baseline / 'future-instruction-interaction-execution'
        assert hashlib.sha256((failed / 'wrapper_failed.json').read_bytes()).hexdigest() == FAILED_SHA
        assert json.loads((failed / 'wrapper_failed.json').read_text())['mode'] == 'simulate'
        assert not (failed / 'complete.json').exists()
        assert not (failed / 'server/progress.json').exists()
        files = [p.relative_to(failed / 'closed-loop').as_posix()
                 for p in (failed / 'closed-loop').rglob('*') if p.is_file()]
        assert files == ['provenance.json'], 'Cannot repeat an executed or partially executed trial'
        frozen, *modules = original_contract(baseline)
        frozen['execution_recovery'] = dict(
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            execution_body_sha256=BODY_SHA, preserved_failed_stage=str(failed),
            preserved_failure_sha256=FAILED_SHA,
            original_failure_before_environment_and_query=True,
            required_environment=dict(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0'),
            scientific_plan_changed=False, q0_recomputed=False)
        return (frozen, *modules)

    body.STAGE = STAGE
    body.contract = contract
    body.main()


if __name__ == '__main__':
    main()
