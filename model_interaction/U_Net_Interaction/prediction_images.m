folder = 'predictions/';
file_name = 'training';
extension = '.TIF';
thresh = 12/255;
location = strcat('200_epochs/',folder);
mkdir(location);
cd Predictions/unet_2400_images_100_epochs_200_fold_unet_2000_images_200_epochs;
files = dir('*.TIF');
for n = 1:length(files)
    file_name = files(n).name;
    image = imread(file_name);
    test_image = imbinarize(image, thresh);
    imwrite(test_image, strcat('../../', location,  file_name));
end
