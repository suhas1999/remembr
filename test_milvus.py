import tempfile, os
from remembr.memory.milvus_memory import MilvusWrapper

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, 'test.db')
    wrapper = MilvusWrapper('test_col', db_path=db_path, drop_collection=False)
    print('MilvusWrapper created OK')
    wrapper.insert([{
        'id': '1',
        'text_embedding': [0.1]*1024,
        'position': [1.0, 2.0, 3.0],
        'theta': 0.5,
        'time': [100.0, 0.0],
        'caption': 'I see a desk',
    }])
    print('Insert OK')
    results = wrapper.search([0.1]*1024, anns_field='text_embedding', limit=1)
    print('Search OK - hits:', len(results[0]))
    print('Caption:', results[0][0]['entity']['caption'])

print('Milvus Lite functional test PASSED')
