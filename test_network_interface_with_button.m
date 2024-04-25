function test_network_interface_with_button(net)
    % Create the main figure
    mainFig = figure('Name', 'Test Network Interface', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600], 'MenuBar', 'none', 'ToolBar', 'none', 'Color', [0.95, 0.95, 0.95]);

    % Create axes for displaying the image
    imageAxes = axes('Parent', mainFig, 'Position', [0.1, 0.3, 0.8, 0.5]);

    % Create the label for displaying the result
    resultLabel = uicontrol('Style', 'text', 'String', '', 'Position', [150, 520, 500, 60], 'HorizontalAlignment', 'center', 'BackgroundColor', [1, 1, 1], 'FontSize', 16, 'FontWeight', 'bold', 'ForegroundColor', [0.2, 0.6, 0.2]);

    % Create the button for selecting an image
    selectImageButton = uicontrol('Style', 'pushbutton', 'String', 'Select Image', 'Position', [100, 30, 150, 40], 'Callback', @selectImageCallback, 'FontSize', 12, 'BackgroundColor', [0.2, 0.5, 0.8], 'ForegroundColor', [1, 1, 1]);

    % Create the button for capturing an image from the camera
    captureImageButton = uicontrol('Style', 'pushbutton', 'String', 'Capture Image', 'Position', [300, 30, 150, 40], 'Callback', @captureImageCallback, 'FontSize', 12, 'BackgroundColor', [0.8, 0.2, 0.2], 'ForegroundColor', [1, 1, 1]);

    % Create the button for detecting multiple people
    detectMultipleButton = uicontrol('Style', 'pushbutton', 'String', 'Detect Multiple', 'Position', [500, 30, 150, 40], 'Callback', @detectMultipleCallback, 'FontSize', 12, 'BackgroundColor', [0.2, 0.8, 0.2], 'ForegroundColor', [1, 1, 1]);

    % Callback function for the "Select Image" button
    function selectImageCallback(~, ~)
        % Open the dialog for selecting an image
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Images (*.jpg,*.png,*.bmp)'}, 'Select an image');

        % If the user cancels the selection, return
        if isequal(filename, 0) || isequal(pathname, 0)
            disp('Image selection canceled.');
            return;
        end

        % Full path of the selected image
        fullpath = fullfile(pathname, filename);

        % Call the test_network function with the selected image
        test_network(net, fullpath, resultLabel, imageAxes);
    end

    % Callback function for the "Capture Image" button
    function captureImageCallback(~, ~)
        % Specify the video format and create a video input object
        video = videoinput('winvideo', 1, 'MJPG_1280x720');

        % Configure the properties of the video input object
        set(video, 'ReturnedColorSpace', 'rgb', 'Timeout', 10);

        % Start the video preview
        preview(video);

        % Attempt to capture a single snapshot from the video
        try
            img = getsnapshot(video);
        catch
            % Display an error message or handle the situation accordingly
            disp('Failed to capture snapshot');
            delete(video);
            return; % Exit the function
        end

        % Stop the video preview
        stoppreview(video);

        % Clean up by deleting the video input object
        delete(video);

        % Save the captured image temporarily
        tempImagePath = tempname + ".png";
        imwrite(img, tempImagePath);

        % Display the snapshot in the imageAxes
        axes(imageAxes);
        imshow(img, 'InitialMagnification', 'fit');
        title('Captured Image', 'FontSize', 18, 'Color', [0.8, 0.2, 0.2]);
        axis off; % Turn off axis for imageAxes

        % Call the test_network function with the captured image
        test_network(net, tempImagePath, resultLabel, imageAxes);

        % Delete the temporary image file
        delete(tempImagePath);
    end

    % Callback function for the "Detect Multiple" button
    function detectMultipleCallback(~, ~)
        % Open the dialog for selecting an image
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Images (*.jpg,*.png,*.bmp)'}, 'Select an image');

        % If the user cancels the selection, return
        if isequal(filename, 0) || isequal(pathname, 0)
            disp('Image selection canceled.');
            return;
        end

        % Full path of the selected image
        fullpath = fullfile(pathname, filename);

        % Display the selected image
        img = imread(fullpath);
        axes(imageAxes);
        imshow(img, 'InitialMagnification', 'fit');
        title('Selected Image', 'FontSize', 18, 'Color', [0.2, 0.6, 0.2]);
        axis off; % Turn off axis for imageAxes

        % Pre-process the image
        gray = rgb2gray(img);

        % Load the pre-trained Haar cascades for detecting people
        classifier = vision.CascadeObjectDetector('ClassificationModel', 'UpperBody', 'MergeThreshold', 16);

        % Detect people in the image
        bodies = step(classifier, gray);

        % Draw bounding boxes around the detected people and count them
        for i = 1:size(bodies, 1)
            rectangle('Position', bodies(i, :), 'EdgeColor', 'g', 'LineWidth', 1);
        end

        % Display the result
        num_people = size(bodies, 1);
        resultString = sprintf('Number of people detected: %d', num_people);
        set(resultLabel, 'String', resultString);
    end
end

% Function for testing the network with an image
function test_network(net, image, resultLabel, imageAxes)
    % Check if the input image is a valid file path
    if ~ischar(image) && ~isstring(image)
        disp('Invalid image path.');
        return;
    end

    % Check if the file exists
    if ~exist(image, 'file')
        disp('Image file not found.');
        return;
    end

    % Read the image
    I = imread(image);

    % Check if the image reading was successful
    if isempty(I)
        disp('Failed to read the image.');
        return;
    end

    G = imresize(I, [224, 224]);

    % Classification of the image
    [label, prob] = classify(net, G);

    % Display the result in the resultLabel
    if max(prob) < 0.8
        resultString = sprintf('Person not detected\nConfidence: %.2f', max(prob));
    else
        resultString = {char(label), ['Confidence: ', num2str(max(prob), 2)]};
    end

    % Set the resultString to the resultLabel
    set(resultLabel, 'String', resultString);

    % Display the image in the specified axes
    axes(imageAxes);
    imshow(G, 'InitialMagnification', 'fit');
    title('Selected Image', 'FontSize', 18, 'Color', [0.2, 0.6, 0.2]);
    axis off; % Turn off axis for imageAxes
end
