import os
import subprocess
import json
import signal
import random
random.seed(42)
import shutil
import time
import re
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from copy import deepcopy
from data_utils import read_jsonl


class TimeoutHandler:
    def __init__(self, timeout, error_message=None):
        self.timeout = timeout
        self.error_message = error_message
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout) #SIGALRM only support unix
        signal.alarm(self.timeout)
    
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
    
    def raise_timeout(self, *args):
        raise TimeoutError(self.error_message)
    

def execute(test_code,timeout=5):
    """try to execute test code"""  
    try:
        exec_globals = {}
        with TimeoutHandler(timeout):
            exec(test_code, globals()) 
            return True
    except AssertionError: #assertionerror is considered as executable
        return True
    except TimeoutError:
        #print("timed out")
        return False
    except Exception as e:
        #print(f"failed: {type(e).__name__}")
        return type(e).__name__, e #return error type and error message
    

def coverage_at_k_sample(passed_tests, k, cov_command_prefix):
    """Compute coverage@k for a single program under test."""
    random.shuffle(passed_tests)
    if len(passed_tests)>=k:
        #num_splits=math.ceil(len(passed_tests)/k) #round up or down?
        num_splits=len(passed_tests)//k
        splited_tests=[passed_tests[i * k : (i + 1) * k] for i in range(num_splits)]
    else: #if number of passed tests is less than k, do not split
        splited_tests=[passed_tests]
    #calculate and average coverages for each group
    split_line_covs=[]
    split_branch_covs=[]
    
    for i,test_group in enumerate(splited_tests):
        group_line_cov=[]
        group_branch_cov=[]
        cov_command=deepcopy(cov_command_prefix)
        for test in test_group:
            cov_command.append(test)
            subprocess.run(cov_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cov_report=json.load(open('coverage.json'))
            total_stmt=cov_report['totals']['num_statements']
            covered_stmt=cov_report['totals']['covered_lines']
            line_cov=covered_stmt/total_stmt
            total_branch=cov_report['totals']['num_branches']
            covered_branch=cov_report['totals']['covered_branches']
            branch_cov=covered_branch/total_branch
            group_line_cov.append(line_cov)
            group_branch_cov.append(branch_cov)
        
        group_avg_line_cov=sum(group_line_cov)/len(group_line_cov)
        group_avg_branch_cov=sum(group_branch_cov)/len(group_branch_cov)
        split_line_covs.append(group_avg_line_cov)
        split_branch_covs.append(group_avg_branch_cov)

    avg_line_cov=sum(split_line_covs)/len(split_line_covs)
    avg_branch_cov=sum(split_branch_covs)/len(split_branch_covs)
    return {'line_cov':avg_line_cov,'branch_cov':avg_branch_cov}
        
    


def check_correctness(generated_data, ks=[1, 2, 5]):
    """Compute syntactical, execution, and assertion correctness (with coverage)."""
    total_cases = 0
    total_syn_correct = 0
    total_comp_correct = 0
    total_exec_correct = 0
    # Add counter for assertion correctness
    total_assert_present = 0
    total_assert_correct = 0
    syn_failed = 0

    exec_fails = []

    total_line_cov = 0
    total_branch_cov = 0
    line_covs_at_k = {f'cov@{k}': [] for k in ks}
    branch_covs_at_k = {f'cov@{k}': [] for k in ks}

    remove_pattern = re.compile(r'tmp*')

    for i, data in tqdm(enumerate(generated_data)):
        task_num = data['task_num']
        difficulty = data['difficulty']
        func_name = data['func_name']
        code = data['code']
        test_cases = data['tests']
        test_import = f'from tmp_{i}_{difficulty}.under_test import Solution\n'
        test_import_simple = f'from under_test import Solution\n'
        os.makedirs(f'tmp_{i}_{difficulty}', exist_ok=True)  # create different tmp folders for different problems
        with open(f'tmp_{i}_{difficulty}/under_test.py', 'w') as f:  # write program under test
            f.write(code)
        passed_tests = []
        
        for j, testcase in enumerate(test_cases):
            total_cases += 1
            
            # Check for assertions
            has_assertion = "assert " in testcase
            if has_assertion:
                total_assert_present += 1
            
            try:
                res = compile(testcase, '<string>', 'exec')  # check syntax correctness
                total_syn_correct += 1

                test_code = test_import + testcase + f'\ntest_{func_name}()'
                time.sleep(0.01)
                
                # Execute test to check if assertion passes
                try:
                    with TimeoutHandler(5):
                        exec(test_code, globals())
                    # If assertion exists and doesn't fail, it's considered correct
                    if has_assertion and test_code.find(f'solution.{func_name}') != -1:
                        total_assert_correct += 1
                except AssertionError:
                    # If assertion fails, it's still an executable test
                    pass
                except:
                    # Other exceptions will be handled by execute()
                    pass
                
                # Regular execution check
                res = execute(test_code)
                if res == True:
                    if test_code.find(f'solution.{func_name}') == -1:  # if function under test is not called
                        print('func under test not called')
                        exec_fails.append({'task': task_num, 'test_num': j, 'error': 'not called'})
                    else:
                        total_exec_correct += 1
                        test_code_simple = test_import_simple + testcase
                        with open(f'tmp_{i}_{difficulty}/test_{j}.py', 'w') as f:
                            f.write(test_code_simple)
                        passed_tests.append(f'test_{j}.py')
                else:
                    exec_fails.append({'task': task_num, 'test_num': j, 'error': res})

            except:
                syn_failed += 1
                pass
        
        if len(passed_tests) > 0:  # start measuring coverage
            # Coverage computation code remains unchanged
            cov_command_prefix = ['pytest', '--cov=under_test', '--cov-branch', '--cov-report=json:coverage.json']
            subprocess.run(f'cp .coveragerc tmp_{i}_{difficulty}/.coveragerc', shell=True)
            os.chdir(f'tmp_{i}_{difficulty}')
            cov_command = deepcopy(cov_command_prefix)
            for test in passed_tests:
                cov_command.append(test)

            try:
                subprocess.run(cov_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cov_report = json.load(open('coverage.json'))
                total_stmt = cov_report['totals']['num_statements']
                covered_stmt = cov_report['totals']['covered_lines']
                line_cov = covered_stmt / total_stmt
                total_branch = cov_report['totals']['num_branches']
                covered_branch = cov_report['totals']['covered_branches']
                branch_cov = covered_branch / total_branch
                total_line_cov += line_cov
                total_branch_cov += branch_cov
            except:
                print('Failed to generate coverage report')
                pass

            # compute coverage@k
            for k in ks:
                res_at_k = coverage_at_k_sample(passed_tests, k, cov_command_prefix)
                line_covs_at_k[f'cov@{k}'].append(res_at_k['line_cov'])
                branch_covs_at_k[f'cov@{k}'].append(res_at_k['branch_cov'])

            os.chdir('..')  # exit tmp_ folder
        else:
            pass

    for dirpath, dirnames, filenames in os.walk('./', topdown=False):
        # Filter dirnames based on the regex pattern
        for dirname in dirnames:
            if remove_pattern.match(dirname):
                shutil.rmtree(dirname)
    
    syn_correct = total_syn_correct / total_cases
    exec_correct = total_exec_correct / total_cases
    # Calculate assertion correctness
    assert_correct = total_assert_correct / total_assert_present if total_assert_present > 0 else 0
    
    all_scores = {
        'syn_correct': syn_correct,
        'exec_correct': exec_correct,
        'assert_correct': assert_correct
    }
    # print(f'Syntax Correctness: {syn_correct}')
    # print(f'Executable Correctness: {exec_correct}')
    # print(f'Assertion Correctness: {assert_correct}')

    # compute average coverage@k
    for k in ks:
        if line_covs_at_k[f'cov@{k}']:
            line_covs_at_k[f'cov@{k}'] = sum(line_covs_at_k[f'cov@{k}']) / len(generated_data)
            branch_covs_at_k[f'cov@{k}'] = sum(branch_covs_at_k[f'cov@{k}']) / len(generated_data)
            # print(f'line coverage@{k}', line_covs_at_k[f'cov@{k}'])
            # print(f'branch coverage@{k}', branch_covs_at_k[f'cov@{k}'])
            all_scores[f'line_covs@{k}'] = line_covs_at_k[f'cov@{k}']
            all_scores[f'branch_covs@{k}'] = branch_covs_at_k[f'cov@{k}']

    # compute coverage
    avg_line_cov = total_line_cov / len(generated_data)
    avg_branch_cov = total_branch_cov / len(generated_data)
    # print(f'Average Line Coverage: {avg_line_cov}, Average Branch Coverage: {avg_branch_cov}')
    all_scores['avg_line_cov', avg_line_cov]
    all_scores['avg_branch_cov', avg_branch_cov]

    # Include assertion correctness in the return value
    return all_scores, exec_fails



def get_reward(generated_data):
    all_scores = check_correctness(generated_data)
    if all_scores['syn_correct'] < 0.98:
        reward = -1.0  # severely penalize if most test cases are invalid
    elif all_scores['exec_correct'] < 0.85:
        reward = 0.0  # discard mostly unexecutable test cases
    else:
        # Weight meaningful test behavior
        reward = (
            0.2 * all_scores['syn_correct'] +
            0.5 * all_scores['exec_correct'] +
            0.3 * all_scores['assert_correct']
        )
        # Bonus for coverage
        if coverage_score is not None:
            reward += 0.5 * all_scores['avg_line_cov'] + 0.5 * all_scores['avg_branch_cov']

    



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='totalcov_gpt-3.5-turbo.jsonl')
    parser.add_argument("--ks", type=int, nargs='+', default=[1, 2, 5])
    return parser.parse_args()


if __name__=='__main__':
    args=parse_args()
    print(args.path)
    print(args.ks)
    output_dir = Path('predictions')
    predictions=read_jsonl(output_dir / args.path)
    print(len(predictions))

    all_scores = check_correctness(predictions, ks=args.ks)
    