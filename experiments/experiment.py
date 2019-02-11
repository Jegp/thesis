"""A module that simplifies running experiments"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import re
import subprocess
import json
import numpy as np
import sys

def compile_snn(dsl, name):
    """Compiles a single dsl expression targetting NEST into a given file"""
    with open(name, 'w') as tmp:
        p = subprocess.run(["volrc", "nest"], input=dsl, stdout=subprocess.PIPE, encoding='utf-8')
        tmp.write(p.stdout)
        
def compile_ann(dsl):
    """Compiles a single dsl expression targetting Futhark into a temporary file, and returns the module"""
    with tempfile.NamedTemporaryFile(suffix='.fut', dir=os.getcwd()) as tmp:
        p = subprocess.run(["volrc"], input=dsl, stdout=subprocess.PIPE, encoding='utf-8')
        tmp.write(p.stdout.encode('utf-8'))
        futhark_file = os.path.basename(tmp.name)
        module_name = futhark_file[:-4]
        tmp.seek(0)
        module_output = os.path.join(os.getcwd(), module_name)
        p = subprocess.run(["futhark", "pyopencl", "--library", futhark_file],
                           stderr=subprocess.PIPE)
        module = importlib.import_module(module_name)
        os.remove(module_name + ".py")
        return getattr(module, module_name)()

REPORT_PATTERN = re.compile(b'\\n(\{.*\})')
def extract_report(output):
    """Helper function to extract reports from experiment stdout"""
    return re.findall(REPORT_PATTERN, output)[0]

def unison_shuffle(a, b):
    """Copies and shuffles two arrays
    Thanks to: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison"""
    permutations = np.random.permutation(len(a))
    return a[permutations], b[permutations]

def create_temp_file(data):
    fp = tempfile.NamedTemporaryFile(mode="w")
    fp.write(str(data.tolist()))
    fp.seek(0)
    return fp

def run_single(experiment_file, xs, ys):
    """Runs a single experiment"""
    data, target = unison_shuffle(xs, ys)
    data_file = create_temp_file(data)
    target_file = create_temp_file(target)
    process = subprocess.Popen(["python3", experiment_file, data_file.name,\
                                target_file.name], 
                                stdout=subprocess.PIPE)
    output = process.stdout.read()
    reportString = extract_report(output)
    data_file.close()
    target_file.close()
    return json.loads(reportString)

def run(experiment_file, xs, ys, iterations=1):
    """Runs an experiment a number of times in parallel"""
    def future_to_result(future):
        try:
            return future.result()
        except Exception as e:
            return "Exception: " + str(e)

    with ThreadPoolExecutor() as pool:
        futures = []
        for index in range(iterations):
            futures.append(pool.submit(run_single, experiment_file, xs, ys))
        return [future_to_result(future) for future in as_completed(futures)]
