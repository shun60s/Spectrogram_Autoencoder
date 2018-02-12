# spectrum
A practice of making mel spectrogram, CNN autoencode pre-training, and classify by deep learning
## Usage
1.making mel spectrogram

Download the english number 0-9 speech data,  <http://pannous.net/files/spoken_numbers_wav.tar>  
(See Description of Data in <https://github.com/AKBoles/Deep-Learning-Speech-Recognition/blob/master/Pannous-Walkthrough.md> )  
and move wav files to wav directory. Steffi were removed due to it may soemthing wrong. 
```
python make_spectrogram.py
```
spectrogram.zip is an example of output spectrogram directory. 
[sample](https://user-images.githubusercontent.com/36104188/36091873-a86aed28-1028-11e8-8e60-0b8a2853c15e.png)

2.making two DataSet for deep learning framework chainer

data data without label (data only) and data set with label 
```
python make_dataset.py
```


## License
 Regarding to melbank.py, follow the license wrtten in the content.

## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
