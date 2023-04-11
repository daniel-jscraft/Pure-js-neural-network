
import * as tf from '@tensorflow/tfjs';
import * as nn from './model';


import {MnistData} from './data';
import * as ui from './ui';
import {argMax} from "./network";

let chart = getChart();

async function train(model, onIteration) {
  ui.logStatus('Training model...');

  const batchSize = 320;

  const validationSplit = 0.15;

  const trainEpochs = ui.getTrainEpochs();

  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData(null);

  const totalNumBatches = Math.ceil(trainData.xs.length / 784 * (1 - validationSplit) / batchSize) * trainEpochs;

  let valAcc = 1;
  await model.train(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: onBatchEnd,
      onEpochEnd: onEpochEnd,
    }
  });

  async function onEpochEnd(epoch, logs) {
    console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss.toFixed(5));
    valAcc = logs.val_acc;
    if (onIteration) {
      onIteration('onEpochEnd', epoch, logs);
    }
      await tf.nextFrame();
  }

  async function onBatchEnd(batch, logs)  {
    trainBatchCount++;
    console.log('onBatchEnd', trainBatchCount, totalNumBatches);
    ui.logStatus(
            `Training... (` +
            `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
            ` complete). To stop training, refresh or close page.`
    );
    if (onIteration && batch % 10 === 0) {
      onIteration('onBatchEnd', batch, logs);
    }
    chartUpdate(logs.loss, trainBatchCount);

    await tf.nextFrame();
  }


  const testAccPercent = model.evaluate(testData.xs, testData.labels);
  ui.logStatus(`Final test accuracy: ${testAccPercent.toFixed(1)}%`
  );
}

async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  tf.tidy(() => {
    const predictions = model.predict(examples.xs);
    let converted = nn.convertFromFlattenArray(examples.xs, examples.labels);
    const labels = convertFromHotEncoding(converted.labels);
    ui.showTestResults(converted.images, predictions, labels);
  });
}

function convertFromHotEncoding(labels) {
  let convLabels = [];
  for (const label of labels) {
    convLabels.push(argMax(label));
  }
  return convLabels;
}


let data;
async function load() {
  data = new MnistData();
  await data.load();
}

ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = nn.getModel(ui.getModelTypeId());
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model, () => showPredictions(model));
});


function getChart() {
  const ctx = document.getElementById('myChart').getContext('2d');
  const myChart = new Chart(ctx, {
    type: 'line',
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true
        }
      }
    },
    data: {
      labels: [],
      datasets: [{
        label: 'Batch Loss',
        data: [],
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
        ],
      }],
    },
  });
  return myChart
}


function chartUpdate(loss, batch) {
  chart.data.labels.push(batch);
  chart.data.datasets.forEach((dataset) => {
    dataset.data.push(loss);
  });
  chart.update();
}
