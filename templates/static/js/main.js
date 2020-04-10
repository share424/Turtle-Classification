function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        
        reader.onload = function(e) {
            $('#gambar').attr('src', e.target.result);
        }
        
        reader.readAsDataURL(input.files[0]); // convert to base64 string
    }
}
$("#file").change(function() {
    readURL(this);
});
async function loadModel() {
    var model = await tf.loadLayersModel('/static/model/model.json');
    return model;
}
$(document).ready(function(e) {
    $(".se-pre-con").fadeOut("slow");
    $(".output").hide();
    $("#form").on('submit', (function(e) {
        e.preventDefault();
        $(".se-pre-con").fadeIn("fast");
        $.ajax({
            url: "preprocess",
            method: "POST",
            contentType: false,
            cache: false,
            processData: false,
            data: new FormData(this),
            success: function(data) {
                loadModel().then(model => {
                    let image_features = data.image_features;
                    let contrast = 0;
                    let correlation = 0;
                    let energy = 0;
                    let homogeneity = 0;
                    for(let i = 0; i<4; i++) {
                        contrast += image_features[i];
                        correlation += image_features[i+4];
                        energy += image_features[i+8];
                        homogeneity += image_features[i+12];
                    }
                    contrast /= 4;
                    correlation /= 4;
                    energy /= 4;
                    homogeneity /= 4;

                    let xs = tf.tensor2d(image_features, [1, 22]);
                    let prediction = model.predict(xs);
                    let labels = ['Nanas Dewasa',
                                    'Tempurung Datar',
                                    'Nanas Muda',
                                    'Baning Coklat',
                                    'Bukan Kura-Kura Bengkulu',
                                    'Pipi Putih',
                                    'Batok',
                                    'Garis Hitam',
                                    'Biuku'];
                    let labelIndex = [0, 3, 0, 4, 7, 5, 2, 1, 6];
                    let predictIndex = prediction.argMax(1).arraySync();
                    let label = labels[predictIndex];
                    let dict = database[predictIndex];

                    d = new Date();
                    $("#clean").attr("src", "/static/uploads/clean.jpg?"+d.getTime());
                    $("#glcm").attr("src", "/static/uploads/glcm.jpg?"+d.getTime());
                    $("#his").attr("src", "/static/uploads/his.jpg?"+d.getTime());
                    $("#source").attr("src", "/static/uploads/image.jpg?"+d.getTime());

                    $("#contrast").html(contrast);
                    $("#correlation").html(correlation);
                    $("#energy").html(energy);
                    $("#homogeneity").html(homogeneity);

                    $("#mean-h").html(image_features[16]);
                    $("#mean-i").html(image_features[17]);
                    $("#mean-s").html(image_features[18]);
                    $("#std-h").html(image_features[19]);
                    $("#std-i").html(image_features[20]);
                    $("#std-s").html(image_features[21]);

                    for(let key in dict) {
                        if(key == 'id') continue;
                        $("#"+key).html(dict[key]);
                    }

                    $(".output").show();
                    $(".se-pre-con").fadeOut("slow");
                });
                
            }
        })
    }));
});