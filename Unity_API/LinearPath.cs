using UnityEngine;
using System.Collections;
using System.IO;

// Place on top of an empty object parenting waypoints. Waypoints can be any objects attached a transform-component
public class LinearPath : MonoBehaviour
{
    #region Public properties
    [Header("Set path to be closed or open")]
    [SerializeField] private bool Closed = false;

    [Header("Set the size of waypoints")]
    [Range(0.0f, 10.0f)]
    [SerializeField] private float waypointSize = 0.1f;

    [Header("Select path colors")]
    [SerializeField] protected Color waypointColor = Color.red;
    [SerializeField] protected Color edgeColor = Color.white;
    #endregion

    #region Private methods
    // Vizualize path in editor by using built-in Gizmos
    private void OnDrawGizmos()
    {
        // Draw all waypoints as spheres
        foreach(Transform t in transform)
        {
            Gizmos.color = waypointColor;
            Gizmos.DrawSphere(t.position, waypointSize);
        }
        
        // Draw lines between all waypoints
        Gizmos.color = edgeColor;
        for (int i = 0; i < transform.childCount - 1; i++)
        {
            Gizmos.DrawLine(transform.GetChild(i).position, transform.GetChild(i+1).position);
        }

        // Close path if desired
        if (Closed)
        {
            Gizmos.DrawLine(transform.GetChild(transform.childCount - 1).position, transform.GetChild(0).position);
        }
    }
    #endregion

    #region Pubic methods
    // Get the next waypoint along the path
    public Transform GetNextWaypoint(Transform currentWaypoint, bool initialize = false)
    {
        // If this is the initial call, get the first waypoint
        if (initialize)
        {
            initialize = false;
            return transform.GetChild(0);
        }
        
        // If this is not the initial or last call, get the next waypoint
        if (currentWaypoint.GetSiblingIndex() < transform.childCount - 1)
        {
            return transform.GetChild(currentWaypoint.GetSiblingIndex() + 1);
        }
        
        // If this is the last call, check if the path is to be closed or not
        else
        {
            // If closed, next waypoint is the first waypoint 
            if (Closed){
                return transform.GetChild(0);
            }

            // If not closed, next waypoint is null
            else
            {
                return null;
            }
        }
    }

    // Write a path to file
    public IEnumerator WritePathToFile(string path, bool append, LinearPath intendedPath)
    {
        // Open a new writer
        StreamWriter writer = new StreamWriter(path, append);

        // Append the transform of each waypoint (child)
        foreach (Transform child in transform){
            // Convert the HP-coordinates to UTM        
            string utm = "";
        
            // Write the child to the supplied writer
            writer.Write(utm + "\n");
        }

        // close the writer
        writer.Close();

        yield return null;
    }
    #endregion
}
