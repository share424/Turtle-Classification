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
                    prediction.print();
                    

                    let labels = ['Nanas Dewasa',
                                    'Tempurung Datar',
                                    'Nanas Muda',
                                    'Baning Coklat',
                                    'Pipi Putih',
                                    'Batok',
                                    'Garis Hitam',
                                    'Biuku'];
                    // let labelIndex = [9, 3, 8, 4, 5, 2, 1, 6];
                    let labelIndex = [4, 2, 6, 1, 9, 8, 5, 3]
                    let predictIndex = prediction.argMax(1).arraySync();
                    //console.log(predictIndex);
                    let confident = prediction.arraySync()[0][predictIndex];
                    let label = labelIndex[predictIndex];
                    if(confident < 0.3) {
                        label = 7;
                    }
                    let dict = database[label];

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
                    
                    $("#confident").html(confident);

                    $(".output").show();
                    $(".se-pre-con").fadeOut("slow");
                });
                
            }
        })
    }));
});