# spectrum
A practice of making mel spectrogram, CNN autoencode pre-training, and classifier by deep learning
## Usage
1.making mel spectrogram

Download the english number 0-9 speech data,  <http://pannous.net/files/spoken_numbers_wav.tar>  
(See Description of Data in <https://github.com/AKBoles/Deep-Learning-Speech-Recognition/blob/master/Pannous-Walkthrough.md> )  
and move wav files to wav directory. Steffi were removed due to it may soemthing wrong. 
```
python make_spectrogram.py
```
Mel scale of FFT size, shift size, bands number, max-min frequecny are adjustable at class GetSpecgram __init__  
Some input wav file of slow utterance (xxx40 <=) or fast utterance (xxx260 >=)  were rejected  
spectrogram.zip is an example of output spectrogram directory.  
![sample](https://user-images.githubusercontent.com/36104188/36091873-a86aed28-1028-11e8-8e60-0b8a2853c15e.png)


2.making DataSet  

each data and label for classifier and autoencoder   
```
python make_dataset.py
```
  

3.classifier by deep learning framework Chainer

 two cnn layers model  
```
python cnn_classifier1-2cnn.py
```
![sample](https://user-images.githubusercontent.com/36104188/36150172-24283bb8-1106-11e8-9bb4-8cccb62466b7.png)

 three cnn layers model  
```
python cnn_classifier1-3cnn.py
```


4. CNN-Autoencoder  
Customized extensions of Updater, Evaluator, and plot_figure are used.  
input->encoder-> decoder ->output   
```
python cnn_autoencoder1.py
```
![sample](https://user-images.githubusercontent.com/36104188/36150167-20d228f2-1106-11e8-9d68-f0a2b217f112.png)


## License
 Regarding to melbank.py, follow the license wrtten in the content.

## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
