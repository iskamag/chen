"""Metadata-only negative controls; never reads labels or starts an engine."""
import argparse
import importlib.util

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--auditor',required=True)
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('mechanical_auditor',args.auditor)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.validate_report_metadata({'worker_exitcodes':[0,0],'wall_seconds':119.960})
for bad in ({'worker_exitcodes':[False,False],'wall_seconds':1.0},
            {'worker_exitcodes':[0,0],'wall_seconds':float('nan')},
            {'worker_exitcodes':[0,0],'wall_seconds':-1.0},
            {'worker_exitcodes':[0,0],'wall_seconds':True}):
    try:module.validate_report_metadata(bad)
    except ValueError:pass
    else:raise AssertionError('negative metadata accepted')
print('PASS valid metadata and four malformed exit/wall negative controls; no label reads')
