from predict_arp import predict_from_csv

results = predict_from_csv('test_live_flow_features.csv')
print(f'Generated {len(results)} predictions')
for r in results:
    print(f"Flow {r['Row']}: {r['Predicted_Category']} (Risk: {r['Risk_Level']}, Confidence: {r['Confidence']})")