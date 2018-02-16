# spectrum
A practice of making mel spectrogram, CNN autoencode pre-training, and classifier by deep learning Chainer  

[github repository](https://github.com/shun60s/spectrum)

## Usage
1.making mel spectrogram

Download the english number 0-9 speech data,  <http://pannous.net/files/spoken_numbers_wav.tar>  
(See Description of Data in <https://github.com/AKBoles/Deep-Learning-Speech-Recognition/blob/master/Pannous-Walkthrough.md> )  
and move wav files to wav directory. Steffi were removed due to it may soemthing wrong. 
```
python make_spectrogram.py
```
Mel scale of FFT size, shift size, bands number, max frequency and min frequecny are adjustable as class GetSpecgram init  
Some input wav file of slow utterance (xxx40 <=) or fast utterance (xxx260 >=)  were rejected  
spectrogram.zip is an example of output spectrogram directory.  
![sample](docs/mel-spectrogram-samples-of-number_0.png)


2.making DataSet

Data set of 2D gray scale image and its label for classifier and autoencoder   
```
python make_dataset.py
```
  

3.classifier by deep learning framework Chainer

 2 CNN layers + FC  model  
```
python cnn_classifier1-2cnn.py
```

 3 CNN cnn layers + FC  model  
```
python cnn_classifier1-3cnn.py
```

![sample](docs/loss-accuracy_comparisonpng.png)  
 2 CNN layers + FC is better performance than 3 CNN layers.  


4.CNN-Autoencoder by deep learning framework Chainer

Customized chainer extensions of Updater, Evaluator, and plot_figure are used.  
input->encoder->decoder->output   
```
python cnn_autoencoder1.py
```
![sample](inout-comparison_autoencoder1-epoch10.png)


input>encoder(fixed)->encoder->decoder>decoder(fxied)->output  
2nd layer of autoencoder training  
```
python cnn_autoencoder2.py
```
![sample](inout-comparison_autoencoder2-epoch10.png)


input>encoder(fixed)->encoder(fixed)>encoder->decoder->decoder(fixed)->decoder(fixed)>output  
3rd layer of autoencoder training  
```
python cnn_autoencoder3.py
```
![sample](inout-comparison_autoencoder3-epoch10.png)


5.classifier with pre-train

load autoencoder trained result, set as initial Weight and bias of CNN, and start training of classifier  
```
python cnn_classifier2-3cnn.py
```
![Comparison](loss-accuracy_comparison_pre-train.png)
With pre-train method rises up faster, but, final performance may depends on layers structure.  


## License
 Regarding to melbank.py, follow the license wrtten in the contents.


## References

- [wav of Pannous, Description](https://github.com/AKBoles/Deep-Learning-Speech-Recognition/blob/master/Pannous-Walkthrough.md)
- [chainer dataset](https://qiita.com/tommyfms2/items/c3fa0cb258c17468cb30)
- [chainer deep autoencoder](https://qiita.com/nyk510/items/bb49e1ab8770f6bfb7d1)
- [chainer extension Evaluator](http://mizti.hatenablog.com/entry/2017/10/24/011003)
- [chainer extension DelGradient](https://qiita.com/ysasaki6023/items/3040fe3896fe1ed844c3)
- [chainer extension GANUpate](https://qiita.com/crcrpar/items/ea05aadeb15aff817546)


## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

