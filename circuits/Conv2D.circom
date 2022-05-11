pragma circom 2.0.3;

include "../node_modules/circomlib-matrix/circuits/matElemMul.circom";
include "../node_modules/circomlib-matrix/circuits/matElemSum.circom";
include "./util.circom";

// Conv2D layer
template Conv2D (nRows, nCols, nChannels, nFilters, kernelSize) {
    signal input in[nRows][nCols][nChannels];
    signal input weights[kernelSize][kernelSize][nChannels][nFilters];
    signal input bias[nFilters];
    signal output out[nRows-kernelSize+1][nCols-kernelSize+1][nFilters];

    component mul[nRows-kernelSize+1][nCols-kernelSize+1][nChannels][nFilters];
    component elemSum[nRows-kernelSize+1][nCols-kernelSize+1][nChannels][nFilters];
    component sum[nRows-kernelSize+1][nCols-kernelSize+1][nFilters];

    for (var i=0; i<nRows-kernelSize+1; i++) {
        for (var j=0; j<nCols-kernelSize+1; j++) {
            for (var k=0; k<nChannels; k++) {
                for (var m=0; m<nFilters; m++) {
                    mul[i][j][k][m] = matElemMul(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            mul[i][j][k][m].a[x][y] <== in[i+x][j+y][k];
                            mul[i][j][k][m].b[x][y] <== weights[x][y][k][m];
                        }
                    }
                    elemSum[i][j][k][m] = matElemSum(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            elemSum[i][j][k][m].a[x][y] <== mul[i][j][k][m].out[x][y];
                        }
                    }
                }
            }
            for (var m=0; m<nFilters; m++) {
                sum[i][j][m] = Sum(nChannels);
                for (var k=0; k<nChannels; k++) {
                    sum[i][j][m].in[k] <== elemSum[i][j][k][m].out;
                }
                out[i][j][m] <== sum[i][j][m].out + bias[m];
            }
        }
    }
}