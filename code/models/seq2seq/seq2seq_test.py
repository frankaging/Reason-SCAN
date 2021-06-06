# TODO: test gSCAN_dataset.py (SOS and EOS and padding and unk)
# TODO: test model.py (masking and stuff)
import unittest
import torch

from seq2seq.model import Model


test_model = Model(input_vocabulary_size=5, embedding_dimension=10, encoder_hidden_size=15,
                   num_encoder_layers=1, target_vocabulary_size=4, encoder_dropout_p=0.,
                   encoder_bidirectional=False, num_decoder_layers=1, decoder_dropout_p=0.,
                   image_dimensions=3, num_cnn_channels=3, cnn_kernel_size=1, cnn_dropout_p=0.,
                   cnn_hidden_num_channels=5, input_padding_idx=0, target_pad_idx=0, target_eos_idx=3,
                   output_directory="test_dir")


class TestGroundedScanDataset(unittest.TestCase):

    def test_situation_encoder(self):
        input_image = torch.zeros()
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == "__main__":
    unittest.main()