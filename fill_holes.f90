program fill
  use fitstools, only: input_map, output_map
  use healpix_types
  use healpix_modules

  implicit none
  integer, parameter :: nside=512, npix = 12*512*512, nmaps=1
  integer, parameter :: new = npix/10
  real(kind=dp) :: mask(0:npix-1,1:1), a(0:npix-1,1:1)
  integer :: i, j, c(0:npix-1,1:1), d(0:npix-1), e(0:npix-1), f(0:npix-1)
  character(len=80), dimension(1:10) :: header

  header(:)=''
  call input_map('partI.fits', mask, npix, nmaps)
  call convert_ring2nest(nside, mask)
  do i = 9,5,-1
    a = mask
    where (a/=dble(i)) c=1
    where (a==dble(i)) c=0
    d(:) = c(:,1)
    call fill_holes_nest(nside, new, d, e)
    f = e-d
    if (i /= 5) then
      where (f==1) mask(:,1)=dble(i-1)
    end if
    d(:) = 1 - d(:)
    call fill_holes_nest(nside, new, d, e)
    f = e-d
    where (f==1) mask(:,1)=dble(i)
  end do
  call convert_nest2ring(nside, mask)
  call output_map(mask, header, 'partIp.fits')
  
  call input_map('partP.fits', mask, npix, nmaps)
  call convert_ring2nest(nside, mask)
  do i = 9,5,-1
    a = mask
    where (a/=dble(i)) c=1
    where (a==dble(i)) c=0
    d(:) = c(:,1)
    call fill_holes_nest(nside, new, d, e)
    f = e-d
    if (i /= 5) then
      where (f==1) mask(:,1)=dble(i-1)
    end if
    d(:) = 1 - d(:)
    call fill_holes_nest(nside, new, d, e)
    f = e-d
    where (f==1) mask(:,1)=dble(i)
  end do
  call convert_nest2ring(nside, mask)
  call output_map(mask, header, 'partPp.fits')
  write(*,*) "hello"





end program fill
