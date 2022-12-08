const chai = require("chai");
const path = require("path");

const wasm_tester = require("circom_tester").wasm;

const F1Field = require("ffjavascript").F1Field;
const Scalar = require("ffjavascript").Scalar;
exports.p = Scalar.fromString("21888242871839275222246405745257275088548364400416034343698204186575808495617");
const Fr = new F1Field(exports.p);

const assert = chai.assert;



describe("Conv2D layer test", function () {
    this.timeout(100000000);

    it("(28,28,1) -> (26,26,4)", async () => {
        let json = require("../models/conv2D_input.json");
        let OUTPUT = require("../models/conv2D_output.json");

        const circuit = await wasm_tester(path.join(__dirname, "circuits", "Conv2D_test.circom"));

        let INPUT = {};

        for (const [key, value] of Object.entries(json)) {
            if (Array.isArray(value)) {
                let tmpArray = [];
                for (let i = 0; i < value.flat().length; i++) {
                    tmpArray.push(Fr.e(value.flat()[i]));
                }
                INPUT[key] = tmpArray;
            } else {
                INPUT[key] = Fr.e(value);
            }
        }

        const witness = await circuit.calculateWitness(INPUT, true);

        assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));

        let ae = 0;

        for (var i=0; i<OUTPUT.out.length; i++) {
            if (OUTPUT.out[i] == 0) {
                ae += Math.abs(OUTPUT.out[i]-parseInt(Fr.toString(witness[i+1])));
            }
            else {
                ae += Math.abs(OUTPUT.out[i]-parseInt(Fr.toString(witness[i+1])));
            }
        }

        const rmae = Math.sqrt(ae/OUTPUT.out.length)*OUTPUT.scale;

        console.log("rmae", rmae);
        assert(rmae < 0.5);

    });

    it("(28,28,1) -> (13,13,4)", async () => {
        let json = require("../models/conv2D_stride_input.json");
        let OUTPUT = require("../models/conv2D_stride_output.json");

        const circuit = await wasm_tester(path.join(__dirname, "circuits", "Conv2D_stride_test.circom"));

        let INPUT = {};

        for (const [key, value] of Object.entries(json)) {
            if (Array.isArray(value)) {
                let tmpArray = [];
                for (let i = 0; i < value.flat().length; i++) {
                    tmpArray.push(Fr.e(value.flat()[i]));
                }
                INPUT[key] = tmpArray;
            } else {
                INPUT[key] = Fr.e(value);
            }
        }

        const witness = await circuit.calculateWitness(INPUT, true);

        assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));

        let ae = 0;

        for (var i=0; i<OUTPUT.out.length; i++) {
            if (OUTPUT.out[i] == 0) {
                ae += Math.abs(OUTPUT.out[i]-parseInt(Fr.toString(witness[i+1])));
            }
            else {
                ae += Math.abs(OUTPUT.out[i]-parseInt(Fr.toString(witness[i+1])));
            }
        }

        const rmae = Math.sqrt(ae/OUTPUT.out.length)*OUTPUT.scale;

        console.log("rmae", rmae);
        assert(rmae < 0.5);

    });
});