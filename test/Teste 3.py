import torch
import unittest

class TestConvertMethod(unittest.TestCase):

    def convert(self, t, convert_to_format, device='cpu', dtype=None, non_blocking=False):
        try:
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                    memory_format=convert_to_format,
                )
            return t.to(
                device,
                dtype if t.is_floating_point() or t.is_complex() else None,
                non_blocking,
            )
        except NotImplementedError as e:
            if str(e) == "Cannot copy out of meta tensor; no data!":
                raise NotImplementedError(
                    f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                    f"when moving module from meta to a different device."
                ) from None
            else:
                raise

    #Casos de Teste 1:Teste (V,V)
    def test_case_1(self):
        t = torch.randn(2, 3, 4, 5)  # Tensor de 4 dimensões
        convert_to_format = torch.contiguous_format
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.shape, t.shape)  # Verifica se o formato foi mantido
    #Testa o caso que o convert não é None e t.dim é 4 ou 5

    #Casos de Teste 2:Teste (V,F)
    def test_case_2(self):
        t = torch.randn(2, 3)  # Tensor de 2 dimensões
        convert_to_format = torch.contiguous_format
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.shape, t.shape)  #Verifica se o formato foi mantido
    #Testa o caso que o convert não é None e t.dim não é 4 ou 5

    #Casos de Teste 3:Teste (F,V)
    def test_case_3(self):
        t = torch.randn(2, 3, 4, 5)  #Tensor de 4 dimensões
        convert_to_format = None
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.shape, t.shape)  #Verifica se o formato foi mantido
    #Testa o caso que o convert é None e t.dim é 4 ou 5

    #Casos de Teste 4:Teste (F,F)
    def test_case_4(self):
        t = torch.randn(2, 3)  #Tensor de 2 dimensões
        convert_to_format = None
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.shape, t.shape)  #Verifica se o formato foi mantido
    #Testa o caso que o convert é None e t.dim não é 4 ou 5

    #Casos de Teste 5:Teste (dado flutuante)
    def test_case_5(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)  #Tensor flutuante
        convert_to_format = None
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.dtype, torch.float32)  #Verifica se é flutuante
    #Testa o tipo de dado

    #Casos de Teste 6:Teste (dado complexo)
    def test_case_6(self):
        t = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)  #Tensor complexo
        convert_to_format = None
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.dtype, torch.complex64)  #Verifica se o tipo é complexo
    #Testa o tipo de dado

    #Casos de Teste 7:Teste (Inteiro)
    def test_case_7(self):
        t = torch.tensor([1, 2], dtype=torch.int32)  #Tensor inteiro
        convert_to_format = None
        result = self.convert(t, convert_to_format)
        self.assertEqual(result.dtype, torch.int32)  #Verifica se o tipo é inteiro
    #Testa o tipo de dado

if __name__ == '__main__':
    unittest.main()
