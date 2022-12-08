pragma circom 2.0.0;

// BatchNormalization layer for 2D inputs
template BatchNormalization2D(nRows, nCols, nChannels) {
    signal input in[nRows][nCols][nChannels];
    signal input zero_point;
    signal output out[nRows][nCols][nChannels];

    for (var i=0; i<nRows; i++) {
        for (var j=0; j<nCols; j++) {
            for (var k=0; k<nChannels; k++) {
                out[i][j][k] <== in[i][j][k]-zero_point;
            }
        }
    }
}