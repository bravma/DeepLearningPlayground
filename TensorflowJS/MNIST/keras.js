const IMAGE_SIZE = 28;

let mnistModel;
const mnist = tf.loadModel("http://localhost:63342/DeepLearningPlayground/TensorflowJS/MNIST/tfjsmodel/model.json").then(function (model) {
    mnistModel = model;
    const image = document.getElementById('img');
    predictImage(image);
});

function predictImage(img) {
    tensor = convertToTensor(img);
    //console.log(tensor);
    let result = mnistModel.predict(tensor);
    result.print();
    //tf.argMax(result).print();
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

$(function(){
    $("#imageUpload").change(function () {
     let imageUpload = document.getElementById("imageUpload");
     if (imageUpload.files.length > 0) {
         $("#txtSelectedFile").val(imageUpload.files[0].name);
     }
     readUrl(this);
    });

    $("#img").on("load", function(){
        console.log("image loaded");
       predictImage(document.getElementById("img"));
    });
});