#include <iostream>
#include <Eigen/Dense>

/*
compile the program with: g++ -I path/to/eigen/ eigen-test.cpp
*/
using namespace std;
using Eigen::MatrixXd;

int main() {
  int rows = 0;
  int columns = 0;
  cout << "please input row number of the matrix: ";
  cin >> rows;
  cout << "please input column number of the matrix: ";
  cin >> columns;
  MatrixXd m(rows, columns);
  for (int i=0; i<rows; ++i) {
    for (int j=0; j<columns; ++j) {
      m(i, j) = i*j;
    }
  }
  cout << m << endl;
}
