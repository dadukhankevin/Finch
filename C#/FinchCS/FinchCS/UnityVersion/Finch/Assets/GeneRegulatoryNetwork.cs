using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class GeneRegulatoryNetwork : MonoBehaviour
{
    
    public string[] Vocab;
    public float[] FloatVocab;
    //if you need these to be foats just divide result
    public int min;
    public int max;

    // Start is called before the first frame update
    void Start()
    {
        
    }
    public IEnumerable<string> genStringVocab(int ammount, bool replace)
    {
        List < string > randomlist = new List<string>(Vocab);        
        List<string> Returnable = new List<string>();
        for (int i = 1; i <= ammount; i++)
        {
            int randomNum = Random.Range(0, randomlist.Count); //pick random index

            Returnable.Add(randomlist[randomNum]);
            if (replace)
            {
                randomlist.RemoveAt(randomNum); //same element can't be chosen more than once
            }
        }
        return Returnable;
    }
    public IEnumerable<float> genFloatVocab(int ammount, bool replace)
    {
        List<float> randomlist = new List<float>(FloatVocab);
        List<float> Returnable = new List<float>();
        for (int i = 1; i <= ammount; i++)
        {
            int randomNum = Random.Range(0, randomlist.Count); //pick random index

            Returnable.Add(randomlist[randomNum]);
            if (replace)
            {
                randomlist.RemoveAt(randomNum); //same element can't be chosen more than once
            }
        }
        return Returnable;
    }
    public IEnumerable<int> genRangeVocab(int ammount)
    {
        List<int> Returnable = new List<int>();
        for (int i = 1; i <= ammount; i++)
        {
            int randomNum = Random.Range(min, max); //pick random index

            Returnable.Add(randomNum);
         
        }
        return Returnable;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
