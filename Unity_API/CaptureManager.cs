using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

public class CaptureManager : MonoBehaviour
{
    #region Private Variables
    [Header("USV transform")]
    [SerializeField] private Transform usv;

    [Header("Camera")]
    [SerializeField] private Camera cam;
    private PerceptionCamera perceptionCam;
    [Tooltip("The delay at each waypoint, giving time for the shutter to capture the terrain")]
    [SerializeField] private float captureDelay = 1.0f;

    [Header("USV intended and actual path")] 
    [Tooltip("Check to update heading by previous waypoint of the actual path. If not, continue along the paths. ")] 
    [SerializeField] private bool updatePath;
    [SerializeField] private LinearPath intendedPath;
    [SerializeField] private LinearPath actualPath;
    [Tooltip("Set the height of the paths manually for ease of debugging/testing")]
    [SerializeField] private float height = 0f;

    [Header("Output dataset")]
    [Tooltip("Filepath to the .txt file containing the waypoints of the intended and actual paths.")]
    [SerializeField] private string filepathWaypoints;

    [Tooltip("Filepath to the .txt file containing the parameters of the intrinsic camera matrix.")]
    [SerializeField] private string filepathCameraMatrix;

    [Tooltip("Filepath to the .txt file containing the raw camera parameters.")]
    [SerializeField] private string filepathCameraParameters;

    private Transform currentWaypoint;
    private Transform currentWaypointIntended;
    private Transform currentWaypointActual;

    private int pathCounter = 0; 

    private bool capturingImage;
    #endregion

    #region Private Methods
    // Start is  automatically called when a script is initialized
    private void Start()
    {   
        // extract the PerceptionCameraComponent from the input camera
        perceptionCam = cam.GetComponent<PerceptionCamera>();

        // update path-heights 
        intendedPath.transform.position = new Vector3(intendedPath.transform.position.x, height, intendedPath.transform.position.z);
        actualPath.transform.position = new Vector3(actualPath.transform.position.x, height, actualPath.transform.position.z);

        // Save paths to .txt file
        StartCoroutine(WritePathsToTxt(filepathWaypoints));

        // Save camera intrinsic matrix to file
        StartCoroutine(WriteMatricesToTxt(filepathCameraMatrix));

        // Save camera parameters to .txt file
        StartCoroutine(WriteCameraParametersToTxt(filepathCameraParameters));

        // start traversing the path
        // NOTE: Have to "request a capture" at waypoint 0. It doesn't get captured, but starts the renderer of labels
        TraversePath(intendedPath, captureImagesAtWaypoint: true, initialize: true);
    }

    // Update is automatically called at each frame
    private void Update()
    {
        // if the usv is not through the intended path
        if(pathCounter < intendedPath.transform.childCount && !capturingImage)
        {   
            if(updatePath)
            {
                TraversePath(intendedPath, captureImagesAtWaypoint: true, initialize: false, updatePrior: true);    
            }
            else
            {
                TraversePath(intendedPath, captureImagesAtWaypoint: true, initialize: false);
            }
        }
        
        // if the usv is just through the intended path, initiate the actual path
        else if(pathCounter == intendedPath.transform.childCount && !capturingImage)
        {
            TraversePath(actualPath, captureImagesAtWaypoint: false, initialize: true);
            pathCounter++;
        }

        // if the usv is not through the actual path
        else if(pathCounter < (intendedPath.transform.childCount + actualPath.transform.childCount) && !capturingImage)
        {
           TraversePath(actualPath, captureImagesAtWaypoint: true, initialize: false);
        }
    }

    // Method to traverse a path from one waypoint to the next. Allowing path initialization and calls to capture-coroutines. 
    private void TraversePath(LinearPath path, bool captureImagesAtWaypoint, bool initialize, bool updatePrior = false)
    {   
        // Get the next waypoint
        currentWaypoint = path.GetNextWaypoint(currentWaypoint, initialize: initialize);
        usv.position = currentWaypoint.position;
        
        // Capture image if requested and rotate
        if(captureImagesAtWaypoint)
        {   
            capturingImage = true;
            StartCoroutine(CaptureImage(path, updatePrior));
            pathCounter++;
        }
        else
        {
            // Look towards the next waypoint
            Transform nextWaypoint = path.GetNextWaypoint(currentWaypoint, initialize: false);
            
            if(nextWaypoint != null)
            {
                usv.LookAt(nextWaypoint);
            }
        }
    }

    // Coroutine to capture images from the USV, allowing delays between captures. 
    private IEnumerator CaptureImage(LinearPath path, bool updatePrior = false)
    {
        // Request a capture from the PerceptionCamera
        perceptionCam.RequestCapture();
        // Induce a delay of 'captureDelay' seconds to allow the rendering and saving of an image
        yield return new WaitForSeconds(captureDelay);
        // Look towards the next waypoint
        Transform nextWaypoint = path.GetNextWaypoint(currentWaypoint, initialize: false);
        
        if(nextWaypoint != null)
        {
            if(updatePrior)
            {
                Transform noisyPosition = actualPath.transform.GetChild(currentWaypoint.GetSiblingIndex());
                usv.position = noisyPosition.position;
            }

            usv.LookAt(nextWaypoint);
        }

        capturingImage = false;
    }

    // Coroutine to write the intended and actual path to a .txt file
    private IEnumerator WritePathsToTxt(string filepath)
    {
        // Open a new writer which overwrites a file at a given path
        StreamWriter writer = new StreamWriter(filepath, false);
        
        // Write the XZ coordinates of the intended waypoints
        foreach (Transform child in intendedPath.transform)
        {
            writer.Write(child.position.x.ToString() + " " + child.position.z.ToString() + "\n");
        }

        // Make sure the actual path children have spawned before trying to write them
        while(actualPath.transform.childCount == 0)
        {
            yield return new WaitForSeconds(3.0f);
        }

        // Write the XZ coordinates of the actual waypoints
        foreach (Transform child in actualPath.transform)
        {
            writer.Write(child.position.x.ToString() + " " + child.position.z.ToString() + "\n");
        }

        // Close the writer
        writer.Close();

        yield return null;
    }

    // Coroutine to write the intended and actual path to a .txt file
    private IEnumerator WriteMatricesToTxt(string filepath)
    {
        // Get camera projection matrices
        Matrix4x4 cameraProjectionMatrix = cam.projectionMatrix;
        Matrix4x4 cameraGPUProjectionMatrix = GL.GetGPUProjectionMatrix(cameraProjectionMatrix, false);

        // Open a new writer which overwrites a file at a given path
        StreamWriter writer = new StreamWriter(filepath, false);
        
        // Write the projection matrix in the same form as a Matrix 4x4
        writer.Write(cameraProjectionMatrix[0, 0].ToString() + " " + cameraProjectionMatrix[0, 1].ToString() + " " + cameraProjectionMatrix[0, 2].ToString() + " " + cameraProjectionMatrix[0, 3].ToString() + "\n");
        writer.Write(cameraProjectionMatrix[1, 0].ToString() + " " + cameraProjectionMatrix[1, 1].ToString() + " " + cameraProjectionMatrix[1, 2].ToString() + " " + cameraProjectionMatrix[1, 3].ToString() + "\n");
        writer.Write(cameraProjectionMatrix[2, 0].ToString() + " " + cameraProjectionMatrix[2, 1].ToString() + " " + cameraProjectionMatrix[2, 2].ToString() + " " + cameraProjectionMatrix[2, 3].ToString() + "\n");
        writer.Write(cameraProjectionMatrix[3, 0].ToString() + " " + cameraProjectionMatrix[3, 1].ToString() + " " + cameraProjectionMatrix[3, 2].ToString() + " " + cameraProjectionMatrix[3, 3].ToString() + "\n");

        // Write the projection matrix in the same form as a Matrix 4x4
        writer.Write(cameraGPUProjectionMatrix[0, 0].ToString() + " " + cameraGPUProjectionMatrix[0, 1].ToString() + " " + cameraGPUProjectionMatrix[0, 2].ToString() + " " + cameraGPUProjectionMatrix[0, 3].ToString() + "\n");
        writer.Write(cameraGPUProjectionMatrix[1, 0].ToString() + " " + cameraGPUProjectionMatrix[1, 1].ToString() + " " + cameraGPUProjectionMatrix[1, 2].ToString() + " " + cameraGPUProjectionMatrix[1, 3].ToString() + "\n");
        writer.Write(cameraGPUProjectionMatrix[2, 0].ToString() + " " + cameraGPUProjectionMatrix[2, 1].ToString() + " " + cameraGPUProjectionMatrix[2, 2].ToString() + " " + cameraGPUProjectionMatrix[2, 3].ToString() + "\n");
        writer.Write(cameraGPUProjectionMatrix[3, 0].ToString() + " " + cameraGPUProjectionMatrix[3, 1].ToString() + " " + cameraGPUProjectionMatrix[3, 2].ToString() + " " + cameraGPUProjectionMatrix[3, 3].ToString() + "\n");


        // Close the writer
        writer.Close();

        yield return null;
    }

    private IEnumerator WriteCameraParametersToTxt(string filepath)
    {
        // Get camera parameters
        float focal_length = cam.focalLength;        // focal length of the camera [mm]
        float sensor_size_x = cam.sensorSize[0];     // size of the camera sensor in x-direction [mm]
        float sensor_size_y = cam.sensorSize[1];     // size of the camera sensor in y-direction [mm]
        float pixel_width = cam.pixelWidth;          // amount of pixels in the width of the camera viewport [px]
        float pixel_height = cam.pixelHeight;        // amount of pixels in the height of the camera viewport [px]
        float principal_point_x = cam.pixelWidth/2;  // principal points x-coordinate from upper left [px]
        float principal_point_y = cam.pixelHeight/2; // principal points y-coordinate from upper left [px]

        // Open a new writer which overwrites a file at a given path
        StreamWriter writer = new StreamWriter(filepath, false);
        
        // Write the projection matrix in the same form as a Matrix 4x4
        writer.Write(focal_length.ToString() + "\n");
        writer.Write(sensor_size_x.ToString() + "\n");
        writer.Write(sensor_size_y.ToString() + "\n");
        writer.Write(pixel_width.ToString() + "\n");
        writer.Write(pixel_height.ToString() + "\n");
        writer.Write(principal_point_x.ToString() + "\n");
        writer.Write(principal_point_y.ToString() + "\n");

        // Close the writer
        writer.Close();

        yield return null;
    }

    #endregion

}
