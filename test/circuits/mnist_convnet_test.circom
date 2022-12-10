pragma circom 2.0.0;

include "../../circuits/Conv2D.circom";
include "../../circuits/Dense.circom";
include "../../circuits/ArgMax.circom";
include "../../circuits/ReLU.circom";
include "../../circuits/MaxPooling2D.circom";
include "../../circuits/Flatten2D.circom";

template mnist_convnet() {
    signal input in[28][28][1];
    signal input conv2d_weights[3][3][1][4];
    signal input conv2d_bias[4];
    signal input conv2d_1_weights[3][3][4][8];
    signal input conv2d_1_bias[8];
    signal input dense_weights[200][10];
    signal input dense_bias[10];
    signal output out;
    signal output dense_out[10];

    component conv2d = Conv2D(28,28,1,4,3,1);
    component re_lu[26][26][4];
    component max_pooling2d = MaxPooling2D(26,26,4,2,2);
    component conv2d_1 = Conv2D(13,13,4,8,3,1);
    component re_lu_1[11][11][8];
    component max_pooling2d_1 = MaxPooling2D(11,11,8,2,2);
    component flatten = Flatten2D(5,5,8);
    component dense = Dense(200,10);
    component argmax = ArgMax(10);

    for (var i=0; i<28; i++) {
        for (var j=0; j<28; j++) {
            conv2d.in[i][j][0] <== in[i][j][0];
        }
    }

    for (var m=0; m<4; m++) {
        for (var i=0; i<3; i++) {
            for (var j=0; j<3; j++) {
                conv2d.weights[i][j][0][m] <== conv2d_weights[i][j][0][m];
            }
        }
        conv2d.bias[m] <== conv2d_bias[m];
    }

    for (var i=0; i<26; i++) {
        for (var j=0; j<26; j++) {
            for (var k=0; k<4; k++) {
                re_lu[i][j][k] = ReLU();
                re_lu[i][j][k].in <== conv2d.out[i][j][k];
                max_pooling2d.in[i][j][k] <== re_lu[i][j][k].out;
            }
        }
    }

    for (var i=0; i<13; i++) {
        for (var j=0; j<13; j++) {
            for (var k=0; k<4; k++) {
                conv2d_1.in[i][j][k] <== max_pooling2d.out[i][j][k];
            }
        }
    }

    for (var m=0; m<8; m++) {
        for (var i=0; i<3; i++) {
            for (var j=0; j<3; j++) {
                for (var k=0; k<4; k++) {
                    conv2d_1.weights[i][j][k][m] <== conv2d_1_weights[i][j][k][m];
                }
            }
        }
        conv2d_1.bias[m] <== conv2d_1_bias[m];
    }

    for (var i=0; i<11; i++) {
        for (var j=0; j<11; j++) {
            for (var k=0; k<8; k++) {
                re_lu_1[i][j][k] = ReLU();
                re_lu_1[i][j][k].in <== conv2d_1.out[i][j][k];
                max_pooling2d_1.in[i][j][k] <== re_lu_1[i][j][k].out;
            }
        }
    }

    for (var i=0; i<5; i++) {
        for (var j=0; j<5; j++) {
            for (var k=0; k<8; k++) {
                flatten.in[i][j][k] <== max_pooling2d_1.out[i][j][k];
            }
        }
    }

    for (var i=0; i<200; i++) {
        dense.in[i] <== flatten.out[i];
        for (var j=0; j<10; j++) {
            dense.weights[i][j] <== dense_weights[i][j];
        }
    }

    for (var i=0; i<10; i++) {
        dense.bias[i] <== dense_bias[i];
    }

    for (var i=0; i<10; i++) {
        dense_out[i] <== dense.out[i];
        argmax.in[i] <== dense.out[i];
    }

    out <== argmax.out;
}

component main = mnist_convnet();