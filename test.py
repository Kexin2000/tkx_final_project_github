import unittest
import os
from app import download_pdf, preprocess, SemanticSearch, generate_text

class TestDownloadPDF(unittest.TestCase):
    def test_download_pdf_success(self):
        # 测试成功下载PDF
        url = "https://huggingface.co/papers/2211.10086"  # 可用的URL
        output_path = "testfile.pdf"
        download_pdf(url, output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_download_pdf_failure(self):
        # 测试下载失败时的处理
        url = "http://hahaha.com/nonexistent.pdf"
        output_path = "nonexistent.pdf"
        with self.assertRaises(Exception):
            download_pdf(url, output_path)

class TestPreprocess(unittest.TestCase):
    def test_preprocess(self):
        text = "This is a test.\nWith some lines."
        processed = preprocess(text)
        self.assertNotIn('\n', processed)
        self.assertRegex(processed, r'\s+')

class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        self.search = SemanticSearch()

    def test_embedding(self):
        data = ["test sentence"]
        self.search.fit(data)
        self.assertEqual(len(self.search.embeddings), 1)

    def test_search_with_no_fit(self):
        # 测试未初始化时的搜索
        self.search = SemanticSearch()
        with self.assertRaises(Exception):
            self.search("test")

class TestGenerateText(unittest.TestCase):
    def test_generate_text(self):
        openAI_key = "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"
        prompt = "Test prompt"
        response = generate_text(openAI_key, prompt)
        self.assertIsNotNone(response)

    def test_generate_text_invalid_key(self):
        # 测试无效的OpenAI key
        openAI_key = "sk-dsojfiojaoidjfklsandfkjioejfioejhiondcknvkjnksdjfk" # 随便编的key
        prompt = "Test prompt"
        with self.assertRaises(Exception):
            generate_text(openAI_key, prompt)

class TestIntegration(unittest.TestCase):
    def test_end_to_end(self):
        # 集成测试：从下载PDF到生成文本
        # 使用有效的URL和OpenAI key
        url = "https://huggingface.co/papers/2211.10086"
        openAI_key = "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"
        download_pdf(url, "integration_test.pdf")
        self.assertTrue(os.path.exists("integration_test.pdf"))
        processed_text = preprocess("Some sample text.")
        self.assertIsNotNone(processed_text)
        response = generate_text(openAI_key, processed_text)
        self.assertIsNotNone(response)



if __name__ == '__main__':
    unittest.main()
