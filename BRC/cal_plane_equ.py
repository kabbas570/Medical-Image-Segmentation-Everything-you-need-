import numpy as np

p1 = np.array([241, 292, 147])
p2 = np.array([242, 292, 148])
p3 = np.array([245, 292, 149])

p_org = np.array([244,292,150])

p4 = np.array([245, 293, 151])
p5 = np.array([246, 291, 151])
p6 = np.array([246, 292, 151])

# Calculate the vector
vector1 = p1 - p6
vector1 = vector1.astype(np.float64)
vector1 /= np.linalg.norm(vector1)
# Calculate the plane 
d = -np.dot(vector1, p_org)
plane_equation = np.append(vector1, d)


# calculate the signed distance of a point from the plane
def signed_distance_from_plane(point, plane_equation):
    return np.dot(plane_equation[:3], point) + plane_equation[3]


import SimpleITK as sitk
image_path = r"C:\My_Data\BRC_Project\D4\D4.seg.nrrd"
img_itk = sitk.ReadImage(image_path)
image_data = sitk.GetArrayFromImage(img_itk)
image_data = np.moveaxis(image_data, [0, 1 ,2], [2,1,0])
line_indices = np.argwhere(image_data == 1)


volume = np.zeros(image_data.shape)

for points in line_indices:
    x,y,z = points[0],points[1],points[2]
    signed_distance = signed_distance_from_plane((x, y, z), plane_equation)

    # Check if voxel lies on the plane, above, or below
    if signed_distance == 0:
        volume[x, y, z] = 1  # Voxel lying on the plane
    elif signed_distance > 0:
        volume[x, y, z] = 2  # Voxel above the plane
    else:
        volume[x, y, z] = -1  # Voxel below the plane


v_above = np.zeros(volume.shape)
v_above[np.where(volume==2)] = 1
v_above = np.moveaxis(v_above,[0, 1 ,2], [2,1,0])
v_above = sitk.GetImageFromArray(v_above)
v_above.CopyInformation(img_itk)
sitk.WriteImage(v_above,r"C:\My_Data\BRC_Project\codes\march5/"+ "v_above.nii.gz")

v_below = np.zeros(volume.shape)
v_below[np.where(volume==-1)] = 1
v_below = np.moveaxis(v_below,[0, 1 ,2], [2,1,0])
v_below = sitk.GetImageFromArray(v_below)
v_below.CopyInformation(img_itk)
sitk.WriteImage(v_below,r"C:\My_Data\BRC_Project\codes\march5/"+ "v_below.nii.gz")













