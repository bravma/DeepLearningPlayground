const buildCnn = function(data) {


    return new Promise(function(resolve, function) {
        const model = tf.sequential();
        model.add(tf.layers.conv1d({
                inputShape: [data.dates.length, 1],
                kernelSize: 100,
                filters: 8,
                stride: 2,
                activation: 'relu',
                kernalInitializer: 'VarianceScaling'
        }));

        model.add(tf.layers.maxPooling1d({
            poolSize: [500],
            stride: 2
        }));

        model.add(tf.layers.conv1d({
            inputShape: [data.dates.length, 1],
            kernelSize: 100,
            filters: 8,
            stride: 2,
            activation: 'relu',
            kernalInitializer: 'VarianceScaling'
        }));

        model.add(tf.layers.maxPooling1d({
            poolSize: [100],
            stride: 2
        }));

        nodel.add(tf.layers.dense({
            units: 10,
            kernalInitializer: 'VarianceScaling',
            activation: 'softmax'
        }));

        return resolve({
            'model': model,
            'data': data
        })
    }
}