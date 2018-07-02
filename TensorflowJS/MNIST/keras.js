const IMAGE_SIZE = 28;

// labels = {}

let mnistModel;
const mnist = tf.loadModel("http://localhost:63342/Python/TensorflowJS/MNIST/tfjsmodel/model.json").then(function (model) {
    mnistModel = model;
    const image = document.getElementById('img');
    predictImage(image);
});

function predictFromTensor(tensor) {
    let logits = mnistModel.predict(tensor);
    logits.data().then(function (data) {
        probabilities = Array.prototype.slice.call(data);
        labelIndex = argMax(probabilities);
        probability = probabilities[labelIndex];
        console.log("Detected number: " + labelIndex +
            " with probability of: " + probability * 100 + " %");

        $("#prediction-label").text(labelIndex);
        $("#prediction-probability").text(probability * 100);
    });
}

function predictImage(img) {
    tensor = convertToTensor(img);
    predictFromTensor(tensor);
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function predictFromCanvas(canv) {
    //get image data
    const ctx = canv.getContext("2d");
    imageData = ctx.getImageData(0, 0, 200, 200);
    //convert to tensor
    tensor = convertToTensor(imageData);
    predictFromTensor(tensor);
}


function convertToTensor(img) {
    if (!(img instanceof tf.Tensor)) {
        img = tf.fromPixels(img, 1);
    }
    let normalized = img.toFloat();
    let resized = normalized;
    if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
    }
    let batched = resized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);
    return batched;
}

function readUrl(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();
        reader.onload = function (e) {

            $("#img").attr("src", e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $("#imageUpload").change(function () {
        let imageUpload = document.getElementById("imageUpload");
        if (imageUpload.files.length > 0) {
            $("#txtSelectedFile").val(imageUpload.files[0].name);
        }
        readUrl(this);
    });

    $("#img").on("load", function () {
        console.log("image loaded");
        predictImage(document.getElementById("img"));
    });

    $("#btnPredictCanvas").click(function () {
        let canvas = $("canvas")[0];
        predictFromCanvas(canvas);
    });

    $("#btnClearCanvas").click(function () {
        let canvas = $("canvas")[0];
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.clearTo("#000000");
    })
});
