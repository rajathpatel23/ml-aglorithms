# include <iostream>
# include <vector>
# include <math.h>
# include <fstream>
# include <string>


using namespace std;

vector <double> expectedOutput;
vector <vector<double>> inputValues;

double weight[] = {0.2, 0.2, 0.2, 0.2};
double learningRate = 0.001;
long epoch = 1000;

double activation(double z);

void updateWeight(double predicted, double expected, vector<double> inputs);
void calculateAccuracy();
void test();

int main() {
    expectedOutput.clear();
    inputValues.clear();

    system("cls");

    string line;

    long i, j;

    ifstream infile;
    infile.open("Iris.csv");

    while (getline(infile, line)) {

        i = 0;
        while(line[i] != ',') {
            i ++;

        }
        i++;

        vector <double> inputRow;
        double output;

        inputRow.clear()

        for(token=0; token < 4; token++){
            double value;
            string val = "";
            while (line[i] != ','){
                val += line[i];

                i ++;
            }
            i++;
            value =  stod(val);
            inputRow.push_back(value);
        }

        i++;

        string outputStr = "";

        outputStr = line[line.size()-1];
        output = stod(outputStr);
        expectedOutput.push_back(output);
        inputValues.push_back(inputRow);
    }

    while (epoch--) {
        calculateAccuracy();
        for (int i = 0; i < inputValues.size(); i++){
            double predictedValue, z = 0;
            for (j = 0,; j < inputValues[0].size(); j++){
                z += weight[j] * inputValues[i][j];
            }
            predictedValue = activation(z);
            updateWeight(predictedValue, expectedOutput[i], inputValues[i]);
        }

    }

    calculateAccuracy();


    return 0;
}

double activation(double z){
    return 1/(1 + pow(e, (-1 * z)));
}

void updateWeight(double predictedValue, double expectedOutput vector<double> inputValue){
    for (int i = 0; i < inputValue.size(); i++){
        double gradientDescent;
        gradientDescent = (predictedValue - expectedOutput) * inputValue[i];
        weight[i] = weight[i] - (gradientDescent * learningRate);
    }
}

void calculateAccuracy(){
    long totalCorrect = 0, totalClass = inputValues.size()
    for (int i = 0; i < inputValues.size(); i++){
        double predictedValue, z = 0;
        for (int j = 0; j < inputValues[i].size(), j++){
            z += inputValues[i][j] * weight[j]
        }

        predictedValue = activation(z)
        if (predictedValue < 0.5){
            predictedValue = 0;
        }
        else {
            predictedValue = 1;
        }

        if predictedValue == expectedOutput[i]{
            totalCorrect ++
        }


    }
    
    cout <<"Accuracy is: "<< (totalCorrect * 100) / totalClass << % endl;
}


void test() {
    double z = 0;
    cout << "Enter the values" << endl;
    for (int i = 0; i < 4; i++){
        double temp;
        cin >> temp;
        z += weight[i] * temp;  
        
    }
    
    double predictedValue = activation(z);

    if (predictedValue < 0.5) {
        cout << "0" << endl;
    }
    else {
        cout << "1" << endl;
    }
}