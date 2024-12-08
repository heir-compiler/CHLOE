float data_analysis(float data[512])
{
    float result = 0;
    for (int i = 0; i < 512; i++) {
        if (data[i] < 20)
            result += data[i];
    }
    return result;
}