'''
Created on 11 févr. 2021

@author: S047432
'''
import unittest
import pathlib
import pprint
from TextProcessor import TextProcessor, TextBox 


class TestTextProcessor(unittest.TestCase):
    '''
    These tests assume tesseract is installed and ready to use with english language
    '''
    
    def _check_box(self, boxes, text_box):
        self.assertTrue(text_box.text in boxes)
        self.assertEqual(boxes[text_box.text], text_box)
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.resource_folder = pathlib.Path(__file__).parent.absolute() / 'resources'

    def test_field_detection(self):
        text_processor = TextProcessor('eng')
        boxes = text_processor.get_text_boxes(str(self.resource_folder / 'form1.jpg'))

        self._check_box(boxes, TextBox(top=31, left=222, width=365, height=22, text='SATISFACTION SURVEY'))
        self._check_box(boxes, TextBox(top=176, left=107, width=185, height=16, text='Enter your personal info'))
        self._check_box(boxes, TextBox(top=222, left=117, width=55, height=9, text='First name'))
        self._check_box(boxes, TextBox(top=272, left=117, width=54, height=8, text='Last name'))
        self.assertEqual(len(boxes), 18)
        
    def test_field_detection2(self):
        text_processor = TextProcessor('eng')
        boxes = text_processor.get_text_boxes(str(self.resource_folder / 'form2.jpg'))

        self._check_box(boxes, TextBox(top=73, left=176, width=180, height=22, text='Registration form'))
        self._check_box(boxes, TextBox(top=125, left=124, width=15, height=12, text='Id:'))
        self._check_box(boxes, TextBox(top=275, left=124, width=94, height=15, text='Email_Address:'))
        self._check_box(boxes, TextBox(top=342, left=124, width=61, height=12, text='About Us:'))
        self.assertEqual(len(boxes), 12)
        
    def test_field_detection3(self):
        text_processor = TextProcessor('eng')
        boxes = text_processor.get_text_boxes(str(self.resource_folder / 'form3.jpg'))

        self._check_box(boxes, TextBox(top=314, left=353, width=178, height=11, text='This is your public name that you'))
        self._check_box(boxes, TextBox(top=329, left=353, width=143, height=11, text='will use to login to the site.'))
        self._check_box(boxes, TextBox(top=622, left=269, width=265, height=11, text='Sign up for our newsletter and find out when new'))
        self.assertEqual(len(boxes), 26)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()