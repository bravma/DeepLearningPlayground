const IMAGE_SIZE = 28;

const mnist = tf.loadModel("http://localhost:63342/Python/TensorflowJS/MNIST/tfjsmodel/model.json").then(function (model) {
    const image = document.getElementById('img');
    tensor = convertToTensor(image);
    console.log(tensor);
    model.predict(tensor).print();
});

function convertToTensor(img) {
    if (!(img instanceof tf.Tensor)) {
        img = tf.fromPixels(img, 1);
    }
    var normalized = img.toFloat();
    var resized = normalized;
    if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        var alignCorners = true;
        resized = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
    }
    var batched = resized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);
    return batched;
}