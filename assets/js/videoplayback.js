$('.videoplay').on('click', function () {
    var vid = $(this).attr('videoid');
    var videoURL = $(vid).prop('src');
    videoURL = videoURL.replace("&autoplay=1", "");
    $(vid).prop('src','');
    $(vid).prop('src',videoURL);
    $(vid)[0].src += "&autoplay=1";
});


$('.videostop').on('click', function () {
    var vid = $(this).attr('videoid');
    var videoURL = $(vid).prop('src');
    videoURL = videoURL.replace("&autoplay=1", "");
    $(vid).prop('src','');
    $(vid).prop('src',videoURL);
});

