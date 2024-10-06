$(document).ready(function() {
    $('#loading').hide(); // Hide the loading spinner initially
    $('#upload-form').submit(function(event) {
        event.preventDefault(); // Prevent form from submitting traditionally
        
        var formData = new FormData();
        var file = $('#video-upload')[0].files[0];
        formData.append('video', file);

        $('#loading').show(); // Show loading spinner
        $('#result').empty(); // Clear previous result

        $.ajax({
            url: '/upload', // Your Flask route for handling the video upload
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#loading').hide(); // Hide loading spinner
                if (response.error) {
                    $('#result').html('<p>Error: ' + response.error + '</p>');
                } else {
                    var resultHTML = '<p>Deepfake: ' + (response.is_deepfake ? 'Yes' : 'No') + '</p>';
                    resultHTML += '<p>Confidence: ' + response.confidence + '</p>';
                    $('#result').html(resultHTML);
                }
            },
            error: function(xhr, status, error) {
                $('#loading').hide(); // Hide loading spinner
                $('#result').html('<p>An error occurred: ' + error + '</p>');
            }
        });
    });
});
