using UnityEngine;
using System.IO;

public class GenerateNoisyLinearPath : LinearPath
{
    #region Public properties
    [Header("The path to add noise upon")]
    [SerializeField] private LinearPath intendedPath;

    [Header("Settings to add noise to the path:")]
    [SerializeField] private dimensionConfig dimension = dimensionConfig.xyz;
    [SerializeField] private float minDeviation = -1.0f;
    [SerializeField] private float maxDeviation = 1.0f;

    #endregion

    #region Private variables

    enum dimensionConfig // your custom enumeration
    {
            xyz, 
            xy, 
            xz
    };

    private Vector3[] noisyPathPoints;
    #endregion

    #region Private methods
    // Generate a random number following a random distribution
    private static float RandomGaussianVariable(float minValue = -1.0f, float maxValue = 1.0f){
        float u, v, S;
        do
        {
            u = 2.0f * UnityEngine.Random.value - 1.0f;
            v = 2.0f * UnityEngine.Random.value - 1.0f;
            S = u * u + v * v;
        }
        while (S >= 1.0f);
    
        // Standard Normal Distribution
        float std = u * Mathf.Sqrt(-2.0f * Mathf.Log(S) / S);
    
        // Normal Distribution centered between the min and max value
        // and clamped following the "three-sigma rule"
        float mean = (minValue + maxValue) / 2.0f;
        float sigma = (maxValue - mean) / 3.0f;
        return Mathf.Clamp(std * sigma + mean, minValue, maxValue);
    }

    // Generate noisy points of a defined path
    private void AddNoiseToPath(LinearPath intendedPath)
    {
        bool firstNode = false;
        foreach (Transform t in intendedPath.transform)
        {
            // Copy position of the original waypoints
            GameObject waypoint = new GameObject("Waypoint" + t);
            waypoint.transform.position = t.position;

            if(!firstNode)
            {
                waypoint.transform.parent = this.gameObject.transform;
                firstNode = true;
                continue;
            }

            // Add noise to the poins in correspondence to the desired dimension
            float deltax, deltay, deltaz;
            if (dimension == dimensionConfig.xyz)
            {
                // Define the noise in xyz
                deltax = deltay = deltaz = RandomGaussianVariable(minDeviation, maxDeviation);
            }
            else if (dimension == dimensionConfig.xy)
            {
                // Define the noise in xy 
                deltax = deltay = RandomGaussianVariable(minDeviation, maxDeviation);
                deltaz = 0.0f;
            }
            else
            {
                // Define the noise in xz 
                deltax = deltaz = RandomGaussianVariable(minDeviation, maxDeviation);
                deltay = 0.0f;
            }

            waypoint.transform.position += new Vector3((int)deltax, (int)deltay, (int)deltaz);

            waypoint.transform.parent = this.gameObject.transform;
        }
    }

    private void Awake()
    {
        // Initialize randomizer
        Random.InitState((int)System.DateTime.Now.Ticks);

        // Generate noisy waypoints
        AddNoiseToPath(intendedPath);
    }
    #endregion

}
