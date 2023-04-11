import {Matrix} from './matrix';

class NeuralNetwork {

    /**
     * @param  {Layer[]} layers - network layers
     */
    constructor(layers) {
        this.layerNodesCounts = []; // no of neurons per layer
        this.layers = layers;
        this.#setLayerNodeCounts(layers);
    }

    #setLayerNodeCounts(layers) {
        for (const layer of layers) {
            if (layer.layerType == Layer.INPUT) {
                continue;
            }
            this.layerNodesCounts.push(layer.weights.cols);
            if (layer.layerType == Layer.OUTPUT) {
                this.layerNodesCounts.push(layer.weights.rows);
            }
        }
    }

    feedForward(input_array, GET_ALL_LAYERS = false) {
        this.#feedforwardArgsValidator(input_array)
        let inputMat = Matrix.fromArray(input_array)
        let outputs = [];
        for (let i = 0; i < this.layerNodesCounts.length; i++) {
            outputs[i] = this.layers[i].processFeedForward(inputMat);
            inputMat = outputs[i];
        }

        if (GET_ALL_LAYERS == true) {
            return outputs;
        }
        return outputs[outputs.length - 1].toArray();
    }


    #feedforwardArgsValidator(input_array) {
        if (input_array.length != this.layers[0].inputs.length) {
            throw new Error("Feedforward failed : Input array and input layer size doesn't match.");
        }
    }
}

export class TrainableNeuralNetwork extends NeuralNetwork {
    learningRate;

    constructor(layers, learningRate = 0.1) {
        super(layers);
        this.learningRate = learningRate;
    }

    fit(input, target) {
        this.#trainArgsValidator(input, target)
        this.feedForward(input, true);
        let loss = this.calculateLoss(target);
        this.updateWeights();
        return loss;
    }

    evaluate(inputs, targets) {
        let total = 0
        for (let i = 0; i < inputs.length; i++) {
            const output = this.predict(inputs[i]);
            total += argMax(output) == argMax(targets[i]) ? 1 : 0;
        }
        return total / inputs.length;
    }

    predict(input) {
        return this.feedForward(input, false);
    }

    calculateLoss(target) {
        const targetMatrix = Matrix.fromArray(target)
        this.#loopLayersInReverse(this.layerNodesCounts, (layerIndex) => {
            let prevLayer
            if (this.layers[layerIndex].layerType != Layer.OUTPUT) {
                prevLayer = this.layers[layerIndex + 1]
            }
            this.layers[layerIndex].calculateErrorLoss(targetMatrix, prevLayer);
        })
        return this.layers[this.layers.length - 1].layerError;
    }

    summary() {
        console.log("Neural Network Summary");
        console.log("Layers : ", this.layerNodesCounts);
        console.log("Learning Rate : ", this.learningRate);
    }

    updateWeights() {
        this.#loopLayersInReverse(this.layerNodesCounts, (layerIndex) => {
            const currentLayer = this.layers[layerIndex]
            const nextLayer = this.layers[layerIndex - 1]
            currentLayer.calculateGradient(this.learningRate);
            currentLayer.updateWeights(nextLayer.outputs);
        })
    }

    #loopLayersInReverse(layerOutputs, callback) {
        for (let layer_index = layerOutputs.length - 1; layer_index >= 1; layer_index--) {
            callback(layer_index)
        }
    }


    #trainArgsValidator(input_array, target_array) {
        if (input_array.length != this.layerNodesCounts[0]) {
            throw new Error("Training failed : Input array and input layer size doesn't match.");
        }
        if (target_array.length != this.layerNodesCounts[this.layerNodesCounts.length - 1]) {
            throw new Error("Training failed : Target array and output layer size doesn't match.");
        }
    }

}

export class Activation {
    static SIGMOID = 1;
    static ReLU = 2;
    static SOFTMAX = 3;

    static create(activationType) {
        switch (activationType) {
            case Activation.SIGMOID:
                return {
                    activation: Activation.#sigmoid,
                    derivative: Activation.#sigmoid_derivative
                }

            case Activation.ReLU:
                return {
                    activation: Activation.#relu,
                    derivative: Activation.#relu_derivative
                }
            case Activation.SOFTMAX:
                return {
                    activation: Activation.#softmax,
                    derivative: Activation.#softmax_derivative
                }
            default:
                console.error('Activation type invalid, setting sigmoid by default');
                return {
                    activation: Activation.sigmoid,
                    derivative: Activation.sigmoid_derivative
                }
        }
    }

    static #softmax_derivative(y) {
        return y * (1 - y);
    }

    static #softmax(x) {
        return 1 / (1 + Math.exp(-x));
    }

    static #sigmoid(x) {
        return 1 / (1 + Math.exp(-1 * x));
    }

    static #sigmoid_derivative(y) {
        return y * (1 - y);
    }

    static #relu(x) {
        if (x >= 0) {
            return x;
        }
        return 0;

    }

    static #relu_derivative(y) {
        if (y > 0) {
            return 1;
        }
        return 0;
    }
}

export class Layer {
    static INPUT = 1
    static HIDDEN = 2
    static OUTPUT = 3
    layerError
    weights

    outputs
    constructor(inputSize, outputSize, activation, layerType) {
        this.layerType = layerType;
        this.activationFun = Activation.create(activation);
        this.weights = Matrix.randomize(outputSize, inputSize);
        this.biases = Matrix.randomize(outputSize, 1);
        this.inputs = new Array(inputSize);
    }

    processFeedForward(input) {
        if (this.layerType == Layer.INPUT) {
            this.inputs = input.data;
            this.outputs = input;
            return input;
        }
        this.inputs = input.data
        let output = Matrix.multiply(this.weights, input);
        output.add(this.biases);
        output.map(this.activationFun.activation);
        this.outputs = output
        return output
    }

    calculateErrorLoss(target_matrix, prevLayer) {
        if (this.layerType == Layer.OUTPUT) {
            this.layerError = Matrix.add(target_matrix, Matrix.multiply(this.outputs, -1));
            return this.layerError;
        }
        const weightTranspose = Matrix.transpose(prevLayer.weights);
        this.layerError = Matrix.multiply(weightTranspose, prevLayer.layerError);
        return this.layerError;
    }

    updateWeights(nextLayerOutput) {
        const nextLayerOutputTransposed = Matrix.transpose(nextLayerOutput);
        const nextWeightsDelta = Matrix.multiply(this.gradient, nextLayerOutputTransposed);

        this.weights.add(nextWeightsDelta);
        this.biases.add(this.gradient);
    }


    calculateGradient(learningRate) {
        this.gradient = Matrix.map(this.outputs, this.activationFun.derivative);
        this.gradient.multiply(this.layerError);
        this.gradient.multiply(learningRate);
    }
}
export function argMax(arr) {
    return arr.indexOf(Math.max(...arr));
}
