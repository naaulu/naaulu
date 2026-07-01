set -e

year=2025
month=07
day=11
country=est
mydate="${year}${month}${day}"
archive="/home/$USER/.cache/naaulu/archive"

first="${mydate}T180500"
last="${mydate}T190000"

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
done

for product in dove eider
do
  naaulu plot \
	 --first $last \
         --country $country \
	 --duration pt1h \
	 --resolution 2km \
	 --product $product \
	 --network est/ghcn 
  dirname="$archive/figure/$year/$month/$day/$country/pt1h/2km/naaulu/$product"
  filename="$first.$country.pt1h.2km.naaulu.$product.png"
  cp "$dirname/$filename" $product.png
done

for product in dove eider
do
  naaulu verify \
	 --first $last \
         --country $country \
	 --duration pt1h \
	 --resolution 2km \
	 --product $product \
	 --network est/ghcn
done
