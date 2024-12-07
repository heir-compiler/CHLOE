LWECipher data_analysis(std::vector<LWECipher> v1) {
  PolyPlain v2 = PolyPlain(MemoryManager::GetPool());
  double v3 = 2.00000000000000000e+01;
  encode_simd(v3, v2);
  RLWECipher v4 = batch_comparison(v1, v2);
  RLWECipher v5 = lwes2rlwe(v1);
  RLWECipher v6 = rlwe_multiply(v4, v5);
  RLWECipher v7 = rlwe_rotate(v6, 256);
  RLWECipher v8 = rlwe_add(v6, v7);
  RLWECipher v9 = rlwe_rotate(v8, 128);
  RLWECipher v10 = rlwe_add(v8, v9);
  RLWECipher v11 = rlwe_rotate(v10, 64);
  RLWECipher v12 = rlwe_add(v10, v11);
  RLWECipher v13 = rlwe_rotate(v12, 32);
  RLWECipher v14 = rlwe_add(v12, v13);
  RLWECipher v15 = rlwe_rotate(v14, 16);
  RLWECipher v16 = rlwe_add(v14, v15);
  RLWECipher v17 = rlwe_rotate(v16, 8);
  RLWECipher v18 = rlwe_add(v16, v17);
  RLWECipher v19 = rlwe_rotate(v18, 4);
  RLWECipher v20 = rlwe_add(v18, v19);
  RLWECipher v21 = rlwe_rotate(v20, 2);
  RLWECipher v22 = rlwe_add(v20, v21);
  RLWECipher v23 = rlwe_rotate(v22, 1);
  RLWECipher v24 = rlwe_add(v22, v23);
  LWECipher v25 = load(v24, 0);
  return v25;
}


