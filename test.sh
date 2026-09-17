set -e
set -x

country=bel
archive="$HOME/.cache/naaulu/archive"
duration="pt10m"

hour=$(date -u -d "1 hour ago" +"%H")
first="$(date -u -d "1 hour ago" +"%Y%m%d")T${hour}4500"
last="$(date -u -d "1 hour ago" +"%Y%m%d")T${hour}5000"

for product in dove eider
do
  naaulu estimate \
	  --first $first \
	  --last $last \
	  --country $country \
	  --product $product
done

for product in dove eider
do
  naaulu combine \
	 --first $last \
         --country $country \
	 --duration $duration \
	 --resolution 2km \
	 --product $product 
done

for product in dove eider
do
  naaulu plot \
	 --first $last \
         --country $country \
	 --duration $duration \
	 --resolution 2km \
	 --product $product \
	 --network $country \
	 --clim 0.1 10
  dirname="$archive/figure/${last:0:4}/${last:4:2}/${last:6:2}/$country/$duration/2km/$product"
  filename="${last//T}.$country.$duration.2km.$product.png"
  cp "$dirname/$filename" $product.png
done

for product in dove eider
do
  naaulu verify \
	 --first $last \
         --country $country \
	 --duration $duration \
	 --resolution 2km \
	 --product $product \
	 --network $country
done

python -c "
import numpy as np; from PIL import Image
a=np.array(Image.open('dove.png').convert('L')).flatten()
b=np.array(Image.open('eider.png').convert('L')).flatten()
corr=np.corrcoef(a,b)[0,1]
print('Correlation:', corr)
assert corr > 0.9, f'Correlation {corr} <= 0.9'
"
