function create_python_detection_code()
    % Specify the Python detection code
    python_code = {
        'import cv2'
        'image = cv2.imread("person_3.png")'
        'hog = cv2.HOGDescriptor()'
        'hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())'
        '[humans, _] = hog.detectMultiScale(image, winStride=(10, 10), padding=(32, 32), scale=1.1)'
        'print("Human Detected :",len(humans))'
        'for (x, y, w, h) in humans:'
        '   pad_w, pad_h = int(0.15 * w), int(0.01 * h)'
        '   cv2.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)'
        'cv2.imshow( "image ", image)'
        'cv2.waitKey(0)'
        'cv2.destroyAllWindows()'
    };

    % Write Python code to file
    fid = fopen('python_detection_code.py', 'w');
    for i = 1:numel(python_code)
        fprintf(fid, '%s\n', python_code{i});
    end
    fclose(fid);

    disp('Python detection code has been created successfully.');
end
