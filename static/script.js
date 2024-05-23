function fileSelected() {
    var fileInput = document.getElementById('videoUpload');
    var file = fileInput.files[0];
    if (file) {
      var fileInfo = document.getElementById('fileInfo');
      fileInfo.innerHTML = 'Selected file: ' + file.name;
      fileInfo.style.display = 'block';
    }
  }
  
  $(document).ready(function() {
    $('#uploadForm').submit(function(e) {
      e.preventDefault();
      var form = $('#uploadForm')[0];
      var formData = new FormData(form);
      
      $('#progressBar').show();
      $('#errorAlert').hide();
      $('#successAlert').hide();
      
      $.ajax({
        xhr: function() {
          var xhr = new window.XMLHttpRequest();
          xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
              var percent = (e.loaded / e.total) * 100;
              $('.progress-bar').css('width', percent + '%');
              $('.progress-bar').attr('aria-valuenow', percent);
              $('.progress-bar').text(Math.round(percent) + '%');
            }
          }, false);
          return xhr;
        },
        type: 'POST',
        url: '/',
        data: formData,
        contentType: false,
        processData: false,
        success: function(response) {
          if (response.error) {
            $('#errorAlert').text(response.error);
            $('#errorAlert').show();
          } else {
            var videoSrc = '/uploads/' + response.uploaded_video;
            $('#uploadedVideoSource').attr('src', videoSrc);
            $('#uploadedVideo')[0].load();
            $('#successAlert').show();
          }
          $('#progressBar').hide();
        },
        error: function(response) {
          $('#errorAlert').text(response.responseJSON.error);
          $('#errorAlert').show();
          $('#progressBar').hide();
        }
      });
    });
  });
