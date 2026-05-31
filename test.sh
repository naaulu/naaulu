set -e

year=2026
month=05
day=18
country=est
mydate="${year}${month}${day}"
archive="/home/$USER/.cache/naaulu/archive"

first="${mydate}T220500"
last="${mydate}T230000"

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
	 --duration pt1h \
	 --resolution 2km \
	 --product $product 
  naaulu plot --duration $duration --first $first --country $country --product $product
  dirname="$archive/figure/$year/$month/$day/$country/pt1h/2km/naaulu/$product"
  filename="$first.$country.pt1h.2km.naaulu.$product.png"
  ls "$dirname/$filename"
done
