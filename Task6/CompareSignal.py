def Compare_Signals(output_file_name, my_file_name):      
    expected_indices=[]
    expected_samples=[]
    my_indices=[]
    my_samples=[]
    with open(output_file_name, 'r') as f:
        line = f.readline()
        # line = f.readline()
        # line = f.readline()
        # line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    with open(my_file_name, 'r') as f:
        line = f.readline()
        # line = f.readline()
        # line = f.readline()
        # line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                my_indices.append(V1)
                my_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(output_file_name)
    print("\n")
    print("My Test file is: ")
    print(my_file_name)
    print("\n")
    if (len(expected_samples)!=len(my_samples)) and (len(expected_indices)!=len(my_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(my_indices)):
        if(my_indices[i]!=expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(my_samples[i] - expected_samples[i]) < 0.25:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one") 
            return
    print("Task 7 Test case passed successfully")

def main():
    """Main function"""
    outputFile = r"D:\University\DSP\Labs\Lab 8\Task 7\FIR test cases\Testcase 1\LPFCoefficients.txt"
    myFile = r"D:\University\DSP\Labs\FrameWork\ir_coefficients.txt"
    Compare_Signals(outputFile, myFile)


if __name__ == "__main__":
    main()
