import SimpleITK as sitk
import numpy as np

itk_image = sitk.ReadImage(r"C:\My_Data\CLIP_WORK\org_data5\FOLDS_Data\F1\SA\val\imgs\004_SA_ES.nii.gz") 


original_spacing = itk_image.GetSpacing()
print(itk_image.GetSize())


import SimpleITK as sitk
import numpy as np

def resample_itk_image(itk_image, out_spacing=(1.0, 1.0, 10.0)):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image



img = resample_itk_image(itk_image)

print(img.GetSpacing())
print(img.GetSize())
