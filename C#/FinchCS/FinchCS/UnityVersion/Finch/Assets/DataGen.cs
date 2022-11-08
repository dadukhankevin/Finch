using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class DataGen : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject individual;
    public int number;
    void Start()
    {
        for (int i = 1; i <= number; i++)
        { // print numbers from 1 to 5
            GameObject IndClone = Instantiate(individual);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
