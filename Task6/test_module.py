"""
Test Module for Time Domain Operations
Contains functions to test all operations with given test cases
"""
import numpy as np
import os


def DerivativeSignal():
    InputSignal = [float(i + 1) for i in range(100)]  # 1 to 100

    # First derivative: y[n] = x[n] - x[n-1]
    FirstDrev = []
    for i in range(1, len(InputSignal)):
        FirstDrev.append(InputSignal[i] - InputSignal[i - 1])

    # Second derivative: y[n] = x[n+1] - 2*x[n] + x[n-1]
    SecondDrev = []
    for i in range(1, len(InputSignal) - 1):
        SecondDrev.append(InputSignal[i + 1] - 2 * InputSignal[i] + InputSignal[i - 1])

    # Expected outputs
    expectedOutput_first = [1] * 99
    expectedOutput_second = [0] * 98

    # Testing
    if (len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second)):
        print("mismatch in length")
        return

    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) > 0.01:
            first = False
            print("1st derivative wrong")
            return

    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) > 0.01:
            second = False
            print("2nd derivative wrong")
            return

    if first and second:
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")


def Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one")
            return
    print("Correlation Test case passed successfully")


def SignalsAreEqual(TaskName, given_output_filePath, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(given_output_filePath, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break

    # FIXED: The return statement was at wrong indentation level
    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        print(TaskName + " Test case failed, your signal have different length from the expected one")
        return  # <-- Now properly inside the if block

    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print(TaskName + " Test case failed, your signal have different indicies from the expected one")
            return

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(TaskName + " Test case failed, your signal have different values from the expected one")
            return

    print(TaskName + " Test case passed successfully")


def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one")
            return
    print("Conv Test case passed successfully")


# ==================================================================

def test_dc_removal():
    """Test DC component removal using provided function"""
    from time_domain_operations import TimeDomainOperations

    print("\n5.Testing DC Component Removal...")

    # Load input signal
    input_path = "cases/DC_component_input.txt"
    if os.path.exists(input_path):
        indices, samples = load_signal_from_file(input_path)
        if indices is not None:
            # Apply DC removal
            dc_removed = TimeDomainOperations.remove_dc(samples)

            # Import and use the provided function
            SignalsAreEqual("DC Removal", "cases/DC_component_output.txt", indices, dc_removed)
    else:
        print(f"  File not found: {input_path}")


def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
    """Test shift and fold operations"""
    return Compare_Signals(file_name, Your_indices, Your_samples)


def load_signal_from_file(file_path):
    """Load signal from standard format file"""
    indices = []
    samples = []

    try:
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')

        if len(lines) < 3:
            return None, None

        signal_type = int(lines[0].strip())
        is_periodic = int(lines[1].strip())
        n_samples = int(lines[2].strip())

        for i in range(3, 3 + n_samples):
            parts = lines[i].strip().split()
            if len(parts) >= 2:
                indices.append(float(parts[0]))
                samples.append(float(parts[1]))

        return np.array(indices), np.array(samples)

    except Exception as e:
        print(f"Error loading signal from {file_path}: {str(e)}")
        return None, None


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("Running Time Domain Operations Tests")
    print("=" * 60)

    # Test 1: Derivatives
    print("\n1.Testing Derivatives...")
    DerivativeSignal()

    # Test 2: Convolution (using hardcoded example)
    print("\n2.Testing Convolution...")
    from time_domain_operations import TimeDomainOperations

    # Example from ConvTest.py
    InputIndicesSignal1 = np.array([-2, -1, 0, 1])
    InputSamplesSignal1 = np.array([1, 2, 1, 1])

    InputIndicesSignal2 = np.array([0, 1, 2, 3, 4, 5])
    InputSamplesSignal2 = np.array([1, -1, 0, 0, 1, 1])

    # Calculate convolution
    y = TimeDomainOperations.convolve_signals(InputSamplesSignal1, InputSamplesSignal2)

    # Generate indices for output
    start_idx = InputIndicesSignal1[0] + InputIndicesSignal2[0]
    Your_indices = np.arange(start_idx, start_idx + len(y))

    ConvTest(Your_indices, y)

    # Test 3: Moving Average (Smoothing)
    print("\n3.Testing Moving Average...")

    # Test 1: Signal1.txt with window size 3
    print("\n**Test 1: Signal1.txt with window size 3")
    signal1_path = "cases/Signal1.txt"
    if os.path.exists(signal1_path):
        indices1, samples1 = load_signal_from_file(signal1_path)
        if indices1 is not None:
            smoothed = TimeDomainOperations.smoothing_moving_average(samples1, 3)
            Compare_Signals("cases/OutMovAvgTest1.txt", indices1[:len(smoothed)], smoothed)
    else:
        print(f"  File not found: {signal1_path}")

    # Test 2: Signal2.txt with window size 5
    print("\n**Test 2: Signal2.txt with window size 5")
    signal2_path = "cases/Signal2.txt"
    if os.path.exists(signal2_path):
        indices2, samples2 = load_signal_from_file(signal2_path)
        if indices2 is not None:
            smoothed = TimeDomainOperations.smoothing_moving_average(samples2, 5)
            Compare_Signals("cases/OutMovAvgTest2.txt", indices2[:len(smoothed)], smoothed)
    else:
        print(f"  File not found: {signal2_path}")

    # ==================================
    # ==================================
    # Test 4: Correlation Tests
    print("\n4.Testing Correlation...")

    # Test correlation for different length signals
    print("\n**Testing correlation for different length signals...")
    signal1_path = "cases/Corr_input_diff_signal1.txt"
    signal2_path = "cases/Corr_input_diff_signal2.txt"

    if os.path.exists(signal1_path) and os.path.exists(signal2_path):
        indices_c1, samples_c1 = load_signal_from_file(signal1_path)
        indices_c2, samples_c2 = load_signal_from_file(signal2_path)

        if indices_c1 is not None and indices_c2 is not None:
            corr = TimeDomainOperations.normalized_cross_correlation(samples_c1, samples_c2)
            # Test file uses indices starting at 0, not actual lag values
            Compare_Signals("cases/CorrDiffOutput.txt", np.arange(len(corr)), corr)
    else:
        print("🚨🚨Correlation test files not found")

    # ++++++++++++++++++++++++++++++++++++++
    # Additional correlation tests (point 1)
    print("\n**Testing correlation for same length signals...")
    signal1b_path = "cases/Corr_input_signal1.txt"
    signal2b_path = "cases/Corr_input_signal2.txt"

    if os.path.exists(signal1b_path) and os.path.exists(signal2b_path):
        indices_c1b, samples_c1b = load_signal_from_file(signal1b_path)
        indices_c2b, samples_c2b = load_signal_from_file(signal2b_path)

        if indices_c1b is not None and indices_c2b is not None:
            corr = TimeDomainOperations.normalized_cross_correlation(samples_c1b, samples_c2b)
            # Test file uses indices starting at 0

            # print(np.arange(len(corr)))
            # print(corr)
            Compare_Signals("cases/CorrOutput.txt", np.arange(len(corr)), corr)
    else:
        print("🚨🚨Point 1 correlation test files not found")

    # Test 5: DC Removal (NEW)
    test_dc_removal()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
