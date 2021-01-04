**Texture mapping using linear transformations**


This project involves the use of geometric transformations in a simple computer vision application. 
The project consists of the following tasks:
1. Transformation library: 
I will create a library for handling general perspective transformations in two dimensions.
2. Affine transformations: 
I will implement code to find an affine transformation between sets of points.
3. Texture mapping: 
I will implement a computer vision application that performs texture mapping using a triangle
mesh. I will use the tranformation library to achieve this.


- Transformation library

The first task is to create a library for creating and combining arbitrary linear transformations in two dimensions and applying them
to two-dimensional points and images. I will use the library in the next task.

Task 1: Implementation

The library uses Numpy for array operations.
It is possible to create the following types of transformations from this library:
Identity, rotation, translation, scaling, arbitrary (i.e. by providing a matrix).
Transformations can be inverted, i.e. I provide a function for performing the inversion.
Transformations can be combined.
Transformations can be applied to two-dimensional points
arranged as a matrix of column vectors. Homographic transformation is handled in the library as well.
Transformations can be applied to images.

Note: I did not use OpenCV for anything except for loading and displaying images.


- Affine estimation

Add a new function learn_affine(points_source, points_target)
to the library that finds the affine transformation between two triangles. Using the matrix inverse method to solve the system.


- Texture mapping

I will now use the library to create a texture mapping application. Specifically, I will be mapping a texture (just an image)
to a triangle mesh in two dimensions. A two-dimensional triangle
mesh is simply a set of triangles that define the shape of a polygon.
Each triangle describes the shape locally.
The shape of the mesh can be changed by moving the points
(typically called vertices) of the mesh. This changes the shape of
each individual triangle in the mesh. Changes to a triangle’s shape can be completely described
by an affine transformation.
The goal of texture mapping is to transform pixels in a texture
according to the shape changes of its associated mesh. This transformation can be described by the transformations of the
individual mesh triangles. I apply the affine shape transformation of a triangle to the texture in the region covered by the triangle.
Each triangle has a position and shape in the original texture.
The shape of the triangle is then changed (in this application by
using the mouse) but the texture points remain stationary. The
texture thus has to be transformed to the shape of the new triangle.
The method can be described as follows. For each triangle in the
mesh the following operations are performed:

1. The affine transformation between the original and deformed triangle is found.
2. The whole image is transformed according to the affine transformation.
3. The image is masked such that only the part covered by the deformed triangle is shown.
4. This triangular image region is added to the final image.
Thus, the final image is created by transforming each image patch according to the triangle deformation.


Task 1: Texture mapping implementation

The application automatically keeps track of the original mesh
points and the deformed mesh points. The mesh is generated automatically from points using cv2.SubDiv2D .

The application is comprised of the following classes:
• TriangleMesh contains all the functionality for creating and modifying the underlying mesh of triangles.
• MeshGUI is responsible for updating the display and handling user events.
• TextureMap is responsible for mapping the texture given a specific TriangleMesh.



Transform individual triangles (Implement transform_patch) :
(a) Find the affine transformation between triangle_base and triangle_transformed using the tranformation library.
(b) Apply the transformation to the entire input texture (available in self.texture ).
(c) Return the transformed image.

Mask the transformed images (Implement mask_patch) :
I transformed the whole image for every triangle and now need to perform the masking step.

(a) Create an empty image for the result using np.zeros() . Use dtype=np.uint8 and the same size as the input patch .
(b) Draw the triangle by using cv2.fillPoly() . Be aware that the function expects a list of arrays of points as input.
(c) Use OpenCV to mask the patch image with the newly created mask.


Update all triangles (Implement (update_patches) :
This method should iterate through each index to be updated and
apply the transformation and masking steps from above.
(a) Iterate over indeces.
(b) For each index, get the original and transformed triangles by calling self.mesh.get_mapping_points(i) .
(c) Apply the transform_patch()
(d) Save the resulting image in self.patches[index].




Perform the texture mapping (Implement get_transformed) :
Using the previously implemented functions, create a complete texture mapping.
(a) Create an empty image to draw individual patches to.
(b) Iterate over self.patches
(c) For each patch, merge it with the empty image using a suitable arithmetic function from OpenCV.
(d) Return the final image.

