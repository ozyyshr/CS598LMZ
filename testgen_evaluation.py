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
import time
import atexit
import functools
# from data_utils import read_jsonl

# pip install pytest pytest-cov

def cleanup_on_exit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        test_dir = None
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Get the test_dir from the function's local variables
            if 'test_dir' in locals():
                test_dir = locals()['test_dir']
            elif len(args) > 3:  # Assuming test_dir is the 4th argument
                test_dir = args[3]
            
            if test_dir and os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir, ignore_errors=True)
                except Exception:
                    pass
    return wrapper

# Register cleanup of any remaining tmp directories on program exit
def cleanup_tmp_dirs():
    try:
        for d in Path('.').glob('tmp_*_test_*'):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

atexit.register(cleanup_tmp_dirs)

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
            print("executed")
            return True
    except AssertionError: #assertionerror is considered as executable
        return True
    except TimeoutError:
        print("timed out")
        return False
    except Exception as e:
        print(f"failed: {type(e).__name__}")
        return type(e).__name__, e #return error type and error message
    

def coverage_at_k_sample(passed_tests, k, cov_command_prefix):
    print(passed_tests, k, cov_command_prefix)
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
    print("-------------------------")
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
        



@cleanup_on_exit
def evaluate_one_case(code, testcase, func_name="solution", i=0, difficulty="test", ks=[1]):
    """
    Evaluate a single test case for a single code snippet.
    
    Args:
        code (str): The code to test
        testcase (str): The test case to evaluate
        func_name (str): Name of the function being tested
        i (int): Index for temporary folder naming
        difficulty (str): Difficulty level for folder naming
        ks (list): List of k values for coverage@k calculation
        
    Returns:
        tuple: (syn_correct, exec_correct, assert_correct, avg_line_cov, avg_branch_cov)
    """
    # Set up environment for testing
    # print(1)
    timestamp = time.time()
    test_dir = f'tmp_{i}_{difficulty}_{str(timestamp).replace(".", "")}'
    os.makedirs(test_dir, exist_ok=True)
    # print(2)
    
    with open(f'{test_dir}/under_test.py', 'w') as f:
        f.write(code)
    
    test_import = f'from {test_dir}.under_test import Solution\n'
    test_import_simple = f'from under_test import Solution\n'
    
    # Initialize results
    syn_correct = 0
    exec_correct = 0
    assert_correct = 0
    assert_present = 0
    line_cov = 0
    branch_cov = 0
    
    # Check for assertions
    has_assertion = "assert " in testcase
    if has_assertion:
        assert_present = 1
    
    # Test syntax correctness
    try:
        compile(testcase, '<string>', 'exec')
        syn_correct = 1
        
        # Prepare full test code
        test_code = test_import + testcase + f'\ntest_{func_name}()'
        
        # Check for assertion correctness
        try:
            with TimeoutHandler(5):
                exec(test_code, globals())
            # If assertion exists and doesn't fail, it's correct
            if has_assertion and test_code.find(f'solution.{func_name}') != -1:
                assert_correct = 1
        except AssertionError:
            # Assertion failed, but test is still executable
            pass
        except Exception:
            # Other exceptions handled by execute()
            pass
        print(test_code)
        # Check execution correctness
        res = execute(test_code)
        # print(2)
        passed_tests = []
        
        if res is True:
            if test_code.find(f'solution.{func_name}') != -1:
                exec_correct = 1
                # Write test to file for coverage measurement
                test_file = f'test_0.py'
                with open(f'{test_dir}/{test_file}', 'w') as f:
                    f.write(test_import_simple + testcase)
                passed_tests.append(test_file)
        
        # Measure coverage if test passed
        if passed_tests:
            cov_command_prefix = ['pytest', '--cov=under_test', '--cov-branch', '--cov-report=json:coverage.json']
            subprocess.run(f'cp .coveragerc {test_dir}/.coveragerc', shell=True)
            
            # Change to test directory
            current_dir = os.getcwd()
            os.chdir(test_dir)
            
            try:
                # Run coverage measurement
                cov_command = deepcopy(cov_command_prefix)
                for test in passed_tests:
                    cov_command.append(test)
                
                subprocess.run(cov_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Parse coverage results
                with open('coverage.json') as f:
                    cov_report = json.load(f)
                
                total_stmt = cov_report['totals']['num_statements']
                covered_stmt = cov_report['totals']['covered_lines']
                line_cov = covered_stmt / total_stmt if total_stmt > 0 else 0
                
                total_branch = cov_report['totals']['num_branches']
                covered_branch = cov_report['totals']['covered_branches']
                branch_cov = covered_branch / total_branch if total_branch > 0 else 0
            except Exception:
                # Failed to generate coverage report
                print(f"Failed to generate coverage report for {test_dir}")
                pass
            finally:
                os.chdir(current_dir)
    except Exception:
        # Syntax error
        pass
    
    if os.path.exists(test_dir):
        try:
            # Return to parent directory in case we're inside the test_dir
            os.chdir(current_dir)
            
            # Force close any open file handles
            import gc
            gc.collect()
            
            # Specifically target .pytest_cache first - often a problematic folder
            pytest_cache_dir = os.path.join(test_dir, '.pytest_cache')
            if os.path.exists(pytest_cache_dir):
                try:
                    # Remove read-only attributes if on Windows
                    if os.name == 'nt':
                        import stat
                        for root, dirs, files in os.walk(pytest_cache_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                os.chmod(file_path, stat.S_IWRITE)
                    
                    # Try to remove .pytest_cache directory specifically
                    shutil.rmtree(pytest_cache_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Error removing pytest cache: {e}")
            
            # Wait a short moment
            time.sleep(0.2)
            
            # Now try to remove the entire test directory
            for attempt in range(3):
                # Try using platform-specific commands first
                if os.name == 'posix':  # Linux/Mac
                    os.system(f"rm -rf {test_dir}")
                elif os.name == 'nt':  # Windows
                    os.system(f"rd /s /q {test_dir}")
                else:
                    shutil.rmtree(test_dir, ignore_errors=True)
                
                # Check if it worked
                if not os.path.exists(test_dir):
                    print(f"Cleaned up test directory: {test_dir}")
                    break
                
                # If still exists, delete any remaining files individually
                if attempt == 1:
                    for root, dirs, files in os.walk(test_dir, topdown=False):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                if os.path.exists(file_path):
                                    os.chmod(file_path, 0o777)  # Set all permissions
                                    os.remove(file_path)
                            except Exception:
                                pass
                        for dir in dirs:
                            try:
                                dir_path = os.path.join(root, dir)
                                if os.path.exists(dir_path):
                                    os.rmdir(dir_path)
                            except Exception:
                                pass
                
                # Wait before trying again
                time.sleep(0.5)
            
            # Final verification
            if os.path.exists(test_dir):
                print(f"Warning: Could not fully remove test directory: {test_dir}")
                # Add this directory to a cleanup list for program exit
                atexit.register(lambda: shutil.rmtree(test_dir, ignore_errors=True) 
                                if os.path.exists(test_dir) else None)
        except Exception as e:
            print(f"Failed to clean up test directory {test_dir}: {str(e)}")
    
    return syn_correct, exec_correct, assert_correct, line_cov, branch_cov




def get_reward(code, testcase, func_name="solution"):
    syn_correct, exec_correct, assert_correct, avg_line_cov, avg_branch_cov = evaluate_one_case(code, testcase, func_name=func_name, ks=[1])
    print("syn_correct: ", syn_correct)
    print("exec_correct: ", exec_correct)
    print("assert_correct: ", assert_correct)
    print("avg_line_cov: ", avg_line_cov)
    print("avg_branch_cov: ", avg_branch_cov)
    if not syn_correct:
        reward = -1  # penalize invalid code
    elif not exec_correct:
        reward = 0  # neutral (valid but fails)
    else:
        reward = 0.2 * assert_correct + 0.4 * avg_line_cov + 0.4 * avg_branch_cov  # bonus for thoroughness
    return reward



def overall_evaluation(generated_data, ks=[1, 2, 5]):
    """Compute syntactical, execution, and assertion correctness (with coverage) using evaluate_one_case."""
    total_cases = 0
    total_syn_correct = 0
    total_exec_correct = 0
    total_assert_present = 0
    total_assert_correct = 0
    
    exec_fails = []
    
    total_line_cov = 0
    total_branch_cov = 0
    
    # Store passed tests for coverage@k calculation
    code_test_mapping = {}
    
    for i, data in tqdm(enumerate(generated_data)):
        task_num = data['task_num']
        difficulty = data['difficulty']
        func_name = data['func_name']
        code = data['code']
        test_cases = data['tests']
        
        code_test_mapping[i] = {
            'code': code,
            'difficulty': difficulty,
            'func_name': func_name,
            'passed_tests': []
        }
        
        for j, testcase in enumerate(test_cases):
            total_cases += 1
            
            # Evaluate this single test case
            syn_correct, exec_correct, assert_correct, line_cov, branch_cov = evaluate_one_case(
                code, testcase, func_name=func_name, i=i, difficulty=difficulty, ks=ks
            )
            
            # Update aggregated metrics
            total_syn_correct += syn_correct
            total_exec_correct += exec_correct
            total_assert_correct += assert_correct
            
            if exec_correct:
                # Store passed test for later coverage@k calculation
                code_test_mapping[i]['passed_tests'].append(testcase)
            else:
                exec_fails.append({
                    'task': task_num,
                    'test_num': j,
                    'error': 'execution failed'
                })
            total_line_cov += line_cov
            total_branch_cov += branch_cov
    
    # Calculate overall metrics
    syn_correct_ratio = total_syn_correct / total_cases if total_cases > 0 else 0
    exec_correct_ratio = total_exec_correct / total_cases if total_cases > 0 else 0
    assert_correct_ratio = total_assert_correct / total_cases if total_cases > 0 else 0
    
    avg_line_cov = total_line_cov / total_cases if total_cases > 0 else 0
    avg_branch_cov = total_branch_cov / total_cases if total_cases > 0 else 0
    
    # Calculate coverage@k metrics
    line_covs_at_k = {f'cov@{k}': [] for k in ks}
    branch_covs_at_k = {f'cov@{k}': [] for k in ks}
    
    for i, data in code_test_mapping.items():
        passed_tests = data['passed_tests']
        
        if len(passed_tests) > 0:
            # Setup for coverage@k calculation
            code = data['code']
            difficulty = data['difficulty']
            func_name = data['func_name']
            test_dir = f'tmp_{i}_{difficulty}'
            
            os.makedirs(test_dir, exist_ok=True)
            with open(f'{test_dir}/under_test.py', 'w') as f:
                f.write(code)
            
            # Write passed tests to files
            test_files = []
            for j, test in enumerate(passed_tests):
                test_file = f'test_{j}.py'
                with open(f'{test_dir}/{test_file}', 'w') as f:
                    f.write(f'from under_test import Solution\n{test}')
                test_files.append(test_file)
            
            # Calculate coverage@k
            cov_command_prefix = ['pytest', '--cov=under_test', '--cov-branch', '--cov-report=json:coverage.json']
            subprocess.run(f'cp .coveragerc {test_dir}/.coveragerc', shell=True)
            current_dir = os.getcwd()
            os.chdir(test_dir)
            
            try:
                for k in ks:
                    if len(test_files) >= k:
                        res_at_k = coverage_at_k_sample(test_files, k, cov_command_prefix)
                        line_covs_at_k[f'cov@{k}'].append(res_at_k['line_cov'])
                        branch_covs_at_k[f'cov@{k}'].append(res_at_k['branch_cov'])
            except Exception:
                pass
            finally:
                os.chdir(current_dir)
                shutil.rmtree(test_dir, ignore_errors=True)
    
    # Compile final results
    all_scores = {
        'syn_correct': syn_correct_ratio,
        'exec_correct': exec_correct_ratio,
        'assert_correct': assert_correct_ratio,
        'avg_line_cov': avg_line_cov,
        'avg_branch_cov': avg_branch_cov
    }
    
    # Add coverage@k metrics
    for k in ks:
        if line_covs_at_k[f'cov@{k}']:
            all_scores[f'line_covs@{k}'] = sum(line_covs_at_k[f'cov@{k}']) / len(generated_data)
            all_scores[f'branch_covs@{k}'] = sum(branch_covs_at_k[f'cov@{k}']) / len(generated_data)
    
    return all_scores, exec_fails





'''

# Sample code implementations
sample_code_1 = """
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
"""

sample_code_2 = """
class Solution:
    def isPalindrome(self, s):
        # Remove non-alphanumeric characters and convert to lowercase
        s = ''.join(c.lower() for c in s if c.isalnum())
        
        # Check if palindrome
        return s == s[::-1]
"""

# Sample test cases
test_case_1_good = """
def test_twoSum():
    solution = Solution()
    # Test case 1
    nums = [2, 7, 11, 15]
    target = 9
    result = solution.twoSum(nums, target)
    assert result == [0, 1] or result == [1, 0]
"""

test_case_1_bad = """
def test_twoSum():
    solution = Solution()
    # Test case with incorrect expected result
    nums = [2, 7, 11, 15]
    target = 9
    result = solution.twoSum(nums, target)
    assert result == [0, 2]  # Incorrect assertion
"""

test_case_2_good = """
def test_isPalindrome():
    solution = Solution()
    # Test case 1
    assert solution.isPalindrome("A man, a plan, a canal: Panama") == True
    
    # Test case 2
    assert solution.isPalindrome("race a car") == False
    
    # Test case 3
    assert solution.isPalindrome("") == True
"""

# Generate sample data for overall_evaluation
sample_generated_data = [
    {
        'task_num': 1,
        'difficulty': 'easy',
        'func_name': 'twoSum',
        'code': sample_code_1,
        'tests': [test_case_1_good, test_case_1_bad]
    },
    {
        'task_num': 2,
        'difficulty': 'easy',
        'func_name': 'isPalindrome',
        'code': sample_code_2,
        'tests': [test_case_2_good]
    }
]

def run_manual_tests():
    """Run manual tests on the code evaluation functions"""
    print("=" * 50)
    print("MANUAL TESTING")
    print("=" * 50)
    
    # Create and save sample data to a file
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save sample data to a JSONL file
    with open(output_dir / 'sample_data.jsonl', 'w') as f:
        for item in sample_generated_data:
            f.write(json.dumps(item) + '\n')
    
    # Test evaluate_one_case with good test case
    print("\nTesting evaluate_one_case with good test case:")
    result = evaluate_one_case(sample_code_1, test_case_1_good, func_name="twoSum")
    print(f"Result: {result}")
    
    # Test evaluate_one_case with bad test case
    print("\nTesting evaluate_one_case with bad test case:")
    result = evaluate_one_case(sample_code_1, test_case_1_bad, func_name="twoSum")
    print(f"Result: {result}")
    
    # Test get_reward with good test case
    print("\nTesting get_reward with good test case:")
    reward = get_reward(sample_code_1, test_case_1_good, func_name="twoSum")
    print(f"Reward: {reward}")
    
    # Test get_reward with bad test case
    print("\nTesting get_reward with bad test case:")
    reward = get_reward(sample_code_1, test_case_1_bad, func_name="twoSum")
    print(f"Reward: {reward}")
    
    # Test overall_evaluation with sample data
    print("\nTesting overall_evaluation with sample data:")
    all_scores, exec_fails = overall_evaluation(sample_generated_data, ks=[1])
    print(f"All scores: {all_scores}")
    print(f"Execution failures: {exec_fails}")
    
    # Clean up
    shutil.rmtree(output_dir)
    
    print("\nManual testing completed.")

if __name__ == "__main__":
    try:
        run_manual_tests()
    except ImportError:
        print("Error: Could not import functions from code_evaluator.py")
        print("Please ensure code_evaluator.py is accessible or adjust the import statements.")
    except Exception as e:
        print(f"Error running manual tests: {e}")

'''











'''
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

    all_scores = overall_evaluation(predictions, ks=args.ks)
'''


if __name__=='__main__':
    test_case = """def test_removeInvalidParentheses():
        solution=Solution()
        test_cases = [
            ("()())()", ["()()"]),
            ("(a)())()", ["(a)()", "(a())"]),
            (")", [""]),
            ("()()(", ["(())", "()()"]),
            ("", [""])
        ]
        for i, (input_str, expected_output) in enumerate(test_cases):
            with self.subTest(i=i):
                assert solution.removeInvalidParentheses(input_str) == expected_output
    """
    func_name = 'removeInvalidParentheses'
    code = """class Solution:
    def removeInvalidParentheses(self, s: str):
            visited = set()
            queue = deque([s])
            result = []
            found = False

            while queue:
                cur = queue.popleft()

                if self.is_valid(cur):
                    found = True
                    result.append(cur)

                if found: continue

                for i in range(len(cur)):
                    if cur[i] == '(' or cur[i] == ')':
                        next_str = cur[:i] + cur[i+1:]
                        if next_str not in visited:
                            visited.add(next_str)
                            queue.append(next_str)

            return result

        def is_valid(self, s: str) -> bool:
            count = 0
            for c in s:
                if c == '(': count += 1
                if c == ')':
                    count -= 1
                    if count < 0: return False
            return count == 0
    """
    
    print(get_reward(code, test_case, func_name))