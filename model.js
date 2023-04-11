import * as tf from '@tensorflow/tfjs';
import {IMAGE_H, IMAGE_W} from "./data";
import {Activation, argMax, Layer, TrainableNeuralNetwork} from "./network";

const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;

function getAvg(losses) {
    return losses
            .map((loss) => Math.abs(loss.data[0][0]))
            .reduce((a, b) => a + b) / losses.length;
}

export function getModel(modelType) {
    let model = createModel(modelType)
    if (model instanceof TrainableNeuralNetwork) {
        return new ModelWrapper(model);
    }
    const optimizer = 'rmsprop';

    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    return new ModelWrapper(model);
}


function createModel(modelType) {
    if (modelType === 'ConvNet') {
        return createConvModel();
    }
    if (modelType === 'DenseNet') {
        return createDenseModel();
    }

    if (modelType === 'MyNet') {
        return createMyModel();
    }
    throw new Error(`Invalid model type: ${modelType}`);
}

export class ModelWrapper {
    /**
     * @param {tf.Sequential| TrainableNeuralNetwork} tfModel
     */
    constructor(tfModel) {
        this.model = tfModel;
    }

    async train(xs, ys, args) {
        if (this.model instanceof tf.Sequential) {
            const inputs = tf.tensor4d(xs, [xs.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
            const labels = tf.tensor2d(ys, [ys.length / NUM_CLASSES, NUM_CLASSES]);
            return this.model.fit(inputs, labels, args)
        }

        return this.#handleMyNNTraining(xs, ys, args);
    }

    #handleMyNNTraining(xs, ys, args) {
        let {images, labels} = convertFromFlattenArray(xs, ys);
        console.log('Training data size: ', images.length)
        console.log("Batch size:", args.batchSize)
        console.log("Epochs:", args.epochs)
        console.log("Validation split:", args.validationSplit)
        let i = 0;
        let epoch = 0;
        let batch = 0;
        const model = this.model;
        return new Promise(function (resolve, reject) {
            let step = () => {
                batch++;
                if (i >= images.length) {
                    i = 0;
                    epoch++;
                    args.callbacks.onEpochEnd(epoch, {val_acc: 1, loss: 0})
                }
                if (epoch + 1 >= args.epochs) {
                    resolve();
                    return;
                }
                let losses = [];
                for (let j = 0; j < args.batchSize; j++) {
                    if (i >= images.length) {
                        console.log("End of epoch " + i)
                        break;
                    }
                    losses.push(model.fit(images[i], labels[i]))
                    i++;
                }
                args.callbacks.onBatchEnd(i, {loss: getAvg(losses) * 100, val_acc: 1})
                requestAnimationFrame(step)
            };
            requestAnimationFrame(step)
        });
    }

    evaluate(xs, ys) {
        if (this.model instanceof tf.Sequential) {
            let inputs = tf.tensor4d(xs, [xs.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
            let labels = tf.tensor2d(ys, [ys.length / NUM_CLASSES, NUM_CLASSES]);
            let testResult = this.model.evaluate(inputs, labels)
            return testResult[1].dataSync()[0] * 100
        }
        let {images, labels} = convertFromFlattenArray(xs, ys);
        return this.model.evaluate(images, labels) * 100
    }

    predict(xs) {
        if (this.model instanceof tf.Sequential) {
            let inputs = tf.tensor4d(xs, [xs.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
            let output = this.model.predict(inputs)

            return Array.from(output.argMax(1).dataSync())
        }
        let {images} = convertFromFlattenArray(xs, null);
        let predictions = []
        for (const image of images) {
            let output = this.model.predict(image);
            predictions.push(argMax(output))
        }
        return predictions
    }

    summary() {
        this.model.summary()
    }
}


function createMyModel() {
    let imageSize = IMAGE_H * IMAGE_W;
    /**
     * @type {Layer[]} layers
     */
    let layers = []
    layers.push(new Layer(
            imageSize,
            imageSize,
            Activation.ReLU,
            Layer.INPUT
    ))
    layers.push(new Layer(
            imageSize,
            42,
            Activation.ReLU,
            Layer.HIDDEN
    ))
    layers.push(new Layer(
            42,
            10,
            Activation.SOFTMAX,
            Layer.OUTPUT
    ))
    return new TrainableNeuralNetwork(layers);
}


function createDenseModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
    model.add(tf.layers.dense({units: 42, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    return model;
}

function createConvModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1],
        kernelSize: 3,
        filters: 16,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

    model.add(tf.layers.flatten({}));

    model.add(tf.layers.dense({units: 64, activation: 'relu'}));

    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

    return model;
}


export function convertFromFlattenArray(xs, ys) {
    const batchSize = xs.length / IMAGE_SIZE;
    let images = [];
    let labels = [];
    for (let i = 0; i < batchSize; i++) {
        const image = xs.slice(i * IMAGE_SIZE, (i + 1) * IMAGE_SIZE);
        images.push(image);

        if (ys) {
            const label = ys.slice(i * NUM_CLASSES, (i + 1) * NUM_CLASSES);
            labels.push(label)
        }
    }
    return {images, labels};
}
