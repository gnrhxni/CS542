#!/usr/bin/perl -w

foreach $fname (@ARGV) {
  if ($fname =~ /^(beta.*r_0..)/) {
     $f2 = $1;
     open(IN,$fname) || die;
     $count{$f2}++;
     open(OUT, ">_${f2}_$count{$f2}") || die;
     while (<IN>) {
         @x = split(/, /);
         printf OUT ("%d %.3f\n", ($x[2]+1)*100, $x[5]);
     }
   }
}
